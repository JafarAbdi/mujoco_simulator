// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fmt/ranges.h>
#include <mujoco/mujoco.h>
#include <mujoco_simulator_msgs/mujoco_simulator.pb.h>
#include <simulate/simulate.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <experimental/array>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <mujoco_simulator/mujoco_model_view.hpp>
#include <mutex>
#include <new>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/to_container.hpp>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <zenoh.hxx>
#include <zenoh/api/bytes.hxx>

#include "array_safety.h"
#include "glfw_adapter.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

#define ASSIGN_OR_RETURN_FALSE(lhs, rhs, ...) \
  auto* lhs = (rhs);                          \
  if (!lhs) {                                 \
    spdlog::error(__VA_ARGS__);               \
    return false;                             \
  }

#define CHECK_OR_RETURN_FALSE(ptr, ...) \
  if (!ptr) {                           \
    spdlog::error(__VA_ARGS__);         \
    return false;                       \
  }

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

/**
 * Serializes a protobuf message into a zenoh::Bytes.
 *
 * @tparam T The protobuf message type (e.g., mujoco_msgs::Person).
 * @param message The protobuf message to serialize.
 * @return A zenoh::Bytes containing the serialized message.
 * @throws std::runtime_error If serialization fails.
 */
template <typename T>
zenoh::Bytes SerializeProtobufToBytes(const T& message) {
  static_assert(std::is_base_of<google::protobuf::Message, T>::value, "T must be a protobuf message type");

  std::vector<uint8_t> buffer(message.ByteSizeLong());

  if (!message.SerializeToArray(buffer.data(), buffer.size())) {
    throw std::runtime_error("Failed to serialize protobuf message");
  }

  return buffer;
}

template <typename T>
T DeserializeProtobufFromBytes(const zenoh::Bytes& buffer) {
  static_assert(std::is_base_of<google::protobuf::Message, T>::value, "T must be a protobuf message type");

  const auto bytes = buffer.as_vector();
  T message;
  if (!message.ParseFromArray(bytes.data(), bytes.size())) {
    throw std::runtime_error("Failed to deserialize protobuf message");
  }
  return message;
}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
const int kErrorLength = 1024;          // load error string length

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;
mjSpec* spec = nullptr;

using Seconds = std::chrono::duration<double>;

// clang-format off
//---------------------------------------- plugin handling -----------------------------------------
// clang-format on

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif

  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

// clang-format off
//------------------------------------------- simulation -------------------------------------------
// clang-format on

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj::Simulate::Clock::now();
  if (mju::strlen_arr(filename) > 4 && !std::strncmp(filename + mju::strlen_arr(filename) - 4,
                                                     ".mjb",
                                                     mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    // Convert to an absolute path, otherwise mujoco complains about duplicate includes.
    // To reproduce run model_act.xml from https://github.com/RainbowRobotics/rby1-sdk/tree/main/models/rby1a/mujoco
    // simulate $(realpath model_act.xml) works
    // simulate model_act.xml fails
    const auto filename_absolute = std::filesystem::absolute(filename);
    spec = mj_parseXML(filename_absolute.c_str(), nullptr, loadError, kErrorLength);
    if (!spec) {
      spdlog::error("Failed to parse spec for file {}: {}",
                    filename_absolute.string(),
                    std::string_view(loadError, kErrorLength));
      exit(1);
    }

    // You have to use spec, otherwise recompiling won't have the qpos from old data
    // See: https://github.com/JafarAbdi/mujoco/tree/spec_bug
    mnew = mj_compile(spec, nullptr);  // mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }
  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  else if (load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);

  return mnew;
}

bool attach_model(const mujoco_simulator_msgs::AttachModelRequest& request, mujoco::Simulate& sim) {
  mjSpec* model_spec = nullptr;
  mjModel* mnew = nullptr;
  mjData* dnew = nullptr;
  {
    std::unique_lock lock(sim.mtx);
    char load_error[kErrorLength] = "";
    model_spec = mj_parseXML(request.model_filename().c_str(), nullptr, load_error, kErrorLength);
    if (!model_spec) {
      spdlog::error("Failed to parse spec: {}", load_error);
      return false;
    }

    ASSIGN_OR_RETURN_FALSE(child_body,
                           mjs_findBody(model_spec, request.child_body_name().c_str()),
                           "Failed to find body: {}",
                           request.child_body_name());

    ASSIGN_OR_RETURN_FALSE(parent_body,
                           mjs_findBody(spec, request.parent_body_name().c_str()),
                           "Failed to find body: {}",
                           request.parent_body_name());

    const auto& pose = request.pose();
    if (!request.has_site_name()) {
      ASSIGN_OR_RETURN_FALSE(
          attachment_site, mjs_addSite(parent_body, nullptr), "Failed to find site: {}", request.site_name());
      mjs_setString(attachment_site->name, request.site_name().c_str());
      mju_copy3(attachment_site->pos, pose.pos().data());
      mju_copy4(attachment_site->quat, pose.quat().data());
      CHECK_OR_RETURN_FALSE(
          mjs_attachToSite(attachment_site, child_body, request.prefix().c_str(), request.suffix().c_str()),
          "Failed to attach body to site");
    } else {
      ASSIGN_OR_RETURN_FALSE(frame, mjs_addFrame(parent_body, nullptr), "Failed to add frame");
      CHECK_OR_RETURN_FALSE(mjs_attachBody(frame, child_body, request.prefix().c_str(), request.suffix().c_str()),
                            "Failed to attach body to frame");
      mju_copy3(frame->pos, pose.pos().data());
      mju_copy4(frame->quat, pose.quat().data());
    }

    mnew = mj_copyModel(nullptr, m);
    dnew = mj_copyData(nullptr, m, d);
    if (mj_recompile(spec, nullptr, mnew, dnew) != 0) {
      spdlog::error("Failed to recompile model");
      return false;
    }

    // TODO(juruc): WTF why mocap pose is set to some crazy random values?? Why is the value from frame-(pos/quat) not
    // used here?
    if (mnew->nmocap > 0) {
      mju_copy3(dnew->mocap_pos, pose.pos().data());
      mju_copy4(dnew->mocap_quat, pose.quat().data());
    }
  }
  sim.Load(mnew, dnew, sim.filename);

  {
    std::unique_lock _(sim.mtx);

    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deleteSpec(model_spec);

    m = mnew;
    d = dnew;
    mj_forward(m, d);
  }

  return true;
}

void load_model(mujoco::Simulate& sim) {
  sim.LoadMessage(sim.filename);
  mjModel* mnew = LoadModel(sim.filename, sim);
  mjData* dnew = nullptr;
  if (mnew) dnew = mj_makeData(mnew);
  if (dnew) {
    sim.Load(mnew, dnew, sim.filename);

    // lock the sim mutex
    const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

    mj_deleteData(d);
    mj_deleteModel(m);

    m = mnew;
    d = dnew;
    mj_forward(m, d);

  } else {
    sim.LoadMessageClear();
  }
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  zenoh::Config config = zenoh::Config::create_default();
  auto session = zenoh::Session::open(std::move(config));

  auto control_subscriber = session.declare_subscriber(zenoh::KeyExpr("robot/ctrl"), zenoh::channels::RingChannel(1));
  auto reset_queryable = session.declare_queryable(
      zenoh::KeyExpr("reset"),
      [&](const zenoh::Query& query) {
        mujoco_simulator_msgs::ResetModelRequest request;
        if (const auto& payload = query.get_payload(); payload.has_value()) {
          const auto bytes = payload->get().as_vector();
          request.ParseFromArray(bytes.data(), bytes.size());
        }
        spdlog::info("Reset model request: {}", request.DebugString());
        if (request.has_model_filename()) {
          mju::strcpy_arr(sim.filename, request.model_filename().c_str());
          load_model(sim);
        }
        {
          std::unique_lock lock(sim.mtx);
          if (request.has_keyframe()) {
            const auto keyframe_names = ModelView(m).keyframe_names();
            const auto keyframe_it = std::ranges::find(keyframe_names, request.keyframe());
            if (keyframe_it == keyframe_names.end()) {
              auto error =
                  fmt::format("Keyframe '{}' not found. Available keyframes: {}", request.keyframe(), keyframe_names);
              spdlog::error(error);
              query.reply(zenoh::KeyExpr("reset"), zenoh::ext::serialize(std::make_tuple(false, std::move(error))));
            }
            const auto keyframe_index = std::distance(keyframe_names.begin(), keyframe_it);
            mj_resetDataKeyframe(m, d, keyframe_index);
          } else {
            mj_resetData(m, d);
          }
          mj_forward(m, d);
          sim.load_error[0] = '\0';
          sim.scrub_index = 0;
          sim.pending_.ui_update_simulation = true;
          query.reply(zenoh::KeyExpr("reset"), zenoh::ext::serialize(std::make_tuple(true, std::string())));
        }
        spdlog::info("Simulation reset.");
        session.put("robot/qpos", zenoh::ext::serialize(std::span(d->qpos, m->nq)));
        session.put("robot/qvel", zenoh::ext::serialize(std::span(d->qvel, m->nv)));
      },
      zenoh::closures::none);

  auto attach_model_queryable = session.declare_queryable(
      zenoh::KeyExpr("attach_model"),
      [&](const zenoh::Query& query) {
        mujoco_simulator_msgs::AttachModelRequest request;
        if (const auto& payload = query.get_payload(); payload.has_value()) {
          const auto bytes = payload->get().as_vector();
          request.ParseFromArray(bytes.data(), bytes.size());
        }
        spdlog::info("Attach model request: {}", request.DebugString());
        mujoco_simulator_msgs::AttachModelResponse response;
        if (!attach_model(request, sim)) {
          response.set_success(false);
          query.reply(zenoh::KeyExpr("attach_model"), SerializeProtobufToBytes(response));
          return;
        }
        response.set_success(true);
        spdlog::info("Model loaded.");
        query.reply(zenoh::KeyExpr("attach_model"), SerializeProtobufToBytes(response));
      },
      zenoh::closures::none);

  auto model_queryable = session.declare_queryable(
      zenoh::KeyExpr("model"),
      [&](const zenoh::Query& query) {
        mujoco_simulator_msgs::GetModelResponse response;
        response.set_model_filename((std::filesystem::current_path() / sim.filename).string());
        query.reply(zenoh::KeyExpr("model"), SerializeProtobufToBytes(response));
      },
      zenoh::closures::none);

  // Should we use a subscriber for this?
  auto remove_geometry_queryable = session.declare_queryable(
      zenoh::KeyExpr("remove_geometry"),
      [&](const zenoh::Query& query) {
        std::unique_lock lock(sim.mtx);
        const auto request = DeserializeProtobufFromBytes<mujoco_simulator_msgs::RemoveVisualGeometryRequest>(
            query.get_payload().value());
        sim.decorative_geoms_.erase(request.name());
        query.reply(zenoh::KeyExpr("remove_geometry"), zenoh::Bytes());
      },
      zenoh::closures::none);

  auto add_geometry_queryable = session.declare_queryable(
      zenoh::KeyExpr("add_geometry"),
      [&](const zenoh::Query& query) {
        std::unique_lock lock(sim.mtx);
        const auto request =
            DeserializeProtobufFromBytes<mujoco_simulator_msgs::AddVisualGeometryRequest>(query.get_payload().value());
        sim.decorative_geoms_.insert_or_assign(request.name(), request.visual_geometry());
        query.reply(zenoh::KeyExpr("add_geometry"), zenoh::Bytes());
      },
      zenoh::closures::none);

  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      load_model(sim);
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than syncmisalign
          bool misaligned = std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(m, d);
            session.put("robot/qpos", zenoh::ext::serialize(std::span(d->qpos, m->nq)));
            session.put("robot/qvel", zenoh::ext::serialize(std::span(d->qvel, m->nv)));
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown = std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise();

              auto result = control_subscriber.handler().try_recv();
              if (std::holds_alternative<zenoh::Sample>(result)) {
                const auto& sample = std::get<zenoh::Sample>(result);

                // TODO: Use protobuf
                const auto control = zenoh::ext::deserialize<std::unordered_map<int, double>>(sample.get_payload());
                ranges::for_each(control, [&](const auto& actuator) { d->ctrl[actuator.first] = actuator.second; });
              }
              // call mj_step
              mj_step(m, d);
              session.put("robot/qpos", zenoh::ext::serialize(std::span(d->qpos, m->nq)));
              session.put("robot/qvel", zenoh::ext::serialize(std::span(d->qvel, m->nv)));
              // TODO: Right now we only send pose for the first mocap
              if (m->nmocap > 0) {
                mujoco_simulator_msgs::Pose pose;
                pose.add_pos(d->mocap_pos[0]);
                pose.add_pos(d->mocap_pos[1]);
                pose.add_pos(d->mocap_pos[2]);
                pose.add_quat(d->mocap_quat[0]);
                pose.add_quat(d->mocap_quat[1]);
                pose.add_quat(d->mocap_quat[2]);
                pose.add_quat(d->mocap_quat[3]);
                session.put("robot/mocap", SerializeProtobufToBytes(pose));
              }
              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}
}  // namespace

// clang-format off
//-------------------------------------- physics_thread --------------------------------------------
// clang-format on

void PhysicsThread(mj::Simulate* sim, const char* filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
    }
    if (d) {
      sim->Load(m, d, filename);

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m, d);

    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteSpec(spec);
  mj_deleteData(d);
  mj_deleteModel(m);
}

// clang-format off
//------------------------------------------ main --------------------------------------------------
// clang-format on

// machinery for replacing command line error by a macOS dialog box when running
// under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char** argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }

  mju_user_error = +[](const char* msg) { spdlog::error("{}", msg); };
  mju_user_warning = +[](const char* msg) { spdlog::warn("{}", msg); };

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // simulate object encapsulates the UI
  auto sim =
      std::make_unique<mj::Simulate>(std::make_unique<mj::GlfwAdapter>(), &cam, &opt, &pert, /* is_passive = */ false);

  const char* filename = nullptr;
  if (argc > 1) {
    filename = argv[1];
  }

  zenoh::init_log_from_env_or("error");

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename);

  // Hide left/right UI by default (TODO: Should we make this a command line option?)
  sim->ui0_enable = 0;
  sim->ui1_enable = 0;

  // start simulation UI loop (blocking call)
  sim->RenderLoop();
  physicsthreadhandle.join();

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
