#include <mujoco/mujoco.h>
#include <spdlog/spdlog.h>

#include <array>

int main(int /*argc*/, char* /*argv*/[]) {
  std::array<char, 1024> error;
  std::array<char, 1024 * 1024> model_str;
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* body = mjs_addBody(world, nullptr);
  mjsSite* site = mjs_addSite(body, nullptr);
  mjModel* model = nullptr;
  mjData* data = nullptr;
  // auto* model = mj_compile(spec, nullptr);
  mj_recompile(spec, nullptr, model, data);
  if (mj_saveXMLString(spec, model_str.data(), model_str.size(), error.data(), error.size()) != 0) {
    spdlog::error("Failed to save model: {}", error.data());
  }
  spdlog::info("Model\n{}", model_str.data());
  return 0;
}
