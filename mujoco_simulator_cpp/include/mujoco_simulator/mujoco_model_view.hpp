#include <fmt/format.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <range/v3/to_container.hpp>
#include <ranges>

template <typename T>
auto make_view(const mjModel* model, int mjModel::*count, T* mjModel::*name_address) {
  // Need to capture by value to avoid dangling references since views are lazy evaluated
  return std::views::iota(0, std::invoke(count, model)) | std::views::transform([=](const auto i) {
           return std::string_view(model->names + std::invoke(name_address, model)[i]);
         });
}

template <>
struct fmt::formatter<mjtJoint> : formatter<std::string_view> {
  auto format(const mjtJoint& joint_type, fmt::format_context& ctx) const -> fmt::format_context::iterator {
    switch (joint_type) {
      case mjtJoint::mjJNT_FREE:
        return fmt::formatter<std::string_view>::format("mjJNT_FREE", ctx);
      case mjtJoint::mjJNT_BALL:
        return fmt::formatter<std::string_view>::format("mjJNT_BALL", ctx);
      case mjtJoint::mjJNT_SLIDE:
        return fmt::formatter<std::string_view>::format("mjJNT_SLIDE", ctx);
      case mjtJoint::mjJNT_HINGE:
        return fmt::formatter<std::string_view>::format("mjJNT_HINGE", ctx);
    }
    return fmt::formatter<std::string_view>::format("Unknown", ctx);
  }
};

struct ModelView {
  const mjModel* model;

  auto body_names() const { return make_view(model, &mjModel::nbody, &mjModel::name_bodyadr); }
  auto joint_names() const { return make_view(model, &mjModel::njnt, &mjModel::name_jntadr); }
  auto geometry_names() const {
    return make_view(model, &mjModel::ngeom, &mjModel::name_geomadr) |
           std::views::filter([this](const auto geometry_name) {
             return mj_name2id(model, mjtObj::mjOBJ_GEOM, geometry_name.data()) != -1;
           });
  }
  auto site_names() const { return make_view(model, &mjModel::nsite, &mjModel::name_siteadr); }
  auto camera_names() const { return make_view(model, &mjModel::ncam, &mjModel::name_camadr); }
  auto light_names() const {
    return make_view(model, &mjModel::nlight, &mjModel::name_lightadr) |
           std::views::filter([this](const auto light_name) {
             return mj_name2id(model, mjtObj::mjOBJ_LIGHT, light_name.data()) != -1;
           });
  }
  auto flex_names() const { return make_view(model, &mjModel::nuser_body, &mjModel::name_flexadr); }
  auto mesh_names() const { return make_view(model, &mjModel::nmesh, &mjModel::name_meshadr); }
  auto actuator_names() const { return make_view(model, &mjModel::nu, &mjModel::name_actuatoradr); }
  auto keyframe_names() const { return make_view(model, &mjModel::nkey, &mjModel::name_keyadr); }
  auto keyframe_qpos() const {
    return std::views::iota(0, model->nkey) | std::views::transform([this](const auto i) {
             return std::span(model->key_qpos + (i * model->nq), model->nq);
           });
  }
  auto keyframe_qvel() const {
    return std::views::iota(0, model->nkey) | std::views::transform([this](const auto i) {
             return std::span(model->key_qvel + (i * model->nv), model->nv);
           });
  }
  // 0 means unlimited, 1 means limited
  auto joint_types() const {
    return std::views::iota(0, model->njnt) |
           std::views::transform([this](const auto i) { return mjtJoint(model->jnt_type[i]); });
  }

  auto joint_limited() const {
    return std::views::iota(0, model->njnt) |
           std::views::transform([this](const auto i) { return model->jnt_limited[i]; });
  }

  auto joint_range() const {
    return std::views::iota(0, model->njnt) |
           std::views::transform([this](const auto i) { return std::span(model->jnt_range + (i * 2), 2); });
  }
};