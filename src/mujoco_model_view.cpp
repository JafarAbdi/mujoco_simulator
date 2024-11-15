#include <fmt/ranges.h>
#include <mujoco/mujoco.h>
#include <spdlog/spdlog.h>

#include <mujoco_simulator/mujoco_model_view.hpp>
#include <range/v3/to_container.hpp>

int main(int argc, char* argv[]) {
  auto* model = mj_loadXML("models/acrobot.xml", nullptr, nullptr, 0);
  mjData* data = mj_makeData(model);

  const auto model_view = ModelView{model};

  spdlog::info("body_names: {}", model_view.body_names());
  spdlog::info("joint_names: {}", model_view.joint_names());
  spdlog::info("geometry_names: {}", model_view.geometry_names());
  spdlog::info("site_names: {}", model_view.site_names());
  spdlog::info("camera_names: {}", model_view.camera_names());
  spdlog::info("light_names: {}", model_view.light_names());
  spdlog::info("flex_names: {}", model_view.flex_names());
  spdlog::info("mesh_names: {}", model_view.mesh_names());
  spdlog::info("actuator_names: {}", model_view.actuator_names());
  spdlog::info("keyframe_names: {}", model_view.keyframe_names());

  // Joints
  spdlog::info("joint_limited: {}", model_view.joint_limited());
  spdlog::info("joint_types: {}", model_view.joint_types());
  spdlog::info("joint_range: {}", model_view.joint_range());

  // Keyframes
  spdlog::info("keyframes qpos: {}", model_view.keyframe_qpos());
  spdlog::info("keyframes qvel: {}", model_view.keyframe_qvel());

  mj_deleteData(data);
  mj_deleteModel(model);
}
