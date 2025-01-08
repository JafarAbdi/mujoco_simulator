#include <fmt/ranges.h>
#include <mujoco/mujoco.h>
#include <spdlog/spdlog.h>

#include <mujoco_simulator/mujoco_model_view.hpp>
#include <range/v3/to_container.hpp>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: {} <model_file>", argv[0]);
    return 1;
  }
  auto* model = mj_loadXML(argv[1], nullptr, nullptr, 0);
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

  // Bodies
  spdlog::info("body_parent_ids: {}", model_view.body_parent_ids());
  spdlog::info("body_root_ids: {}", model_view.body_root_ids());
  spdlog::info("body_tree_ids: {}", model_view.body_tree_ids());
  spdlog::info("body_joint_axes: {}", model_view.body_joint_axes());
  spdlog::info("body_joint_types: {}", model_view.body_joint_types());

  // Joints
  spdlog::info("joint_types: {}", model_view.joint_types());
  spdlog::info("joint_body_ids: {}", model_view.joint_body_ids());
  spdlog::info("joint_limited: {}", model_view.joint_limited());
  spdlog::info("joint_axes: {}", model_view.joint_axes());
  spdlog::info("joint_range: {}", model_view.joint_ranges());

  // Keyframes
  spdlog::info("keyframes qpos: {}", model_view.keyframe_qpos());
  spdlog::info("keyframes qvel: {}", model_view.keyframe_qvel());

  mj_deleteData(data);
  mj_deleteModel(model);
}
