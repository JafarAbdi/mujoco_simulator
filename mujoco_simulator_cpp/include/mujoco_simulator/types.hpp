#include <mujoco/mjtnum.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <zenoh.hxx>

struct AttachModelRequst {
  std::string model_filename;
  std::string parent_body_name;
  std::string child_body_name;  // Not using when site_name is provided
  std::string site_name;        // Will create a site (Not using when frame_name is provided)
  std::array<mjtNum, 3> pos;
  std::array<mjtNum, 4> quat;
  std::string prefix;
  std::string suffix;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    AttachModelRequst, model_filename, parent_body_name, child_body_name, site_name, pos, quat, prefix, suffix);

namespace zenoh::ext::detail {
template <>
inline bool serialize_with_serializer<nlohmann::json>(zenoh::ext::Serializer& serializer,
                                                      const nlohmann::json& t,
                                                      zenoh::ZResult* err) {
  return __zenoh_serialize_with_serializer(serializer, nlohmann::to_string(t), err);
}

template <>
inline bool deserialize_with_deserializer<nlohmann::json>(zenoh::ext::Deserializer& deserializer,
                                                          nlohmann::json& t,
                                                          zenoh::ZResult* err) {
  std::string str;
  if (!__zenoh_deserialize_with_deserializer(deserializer, str, err)) {
    return false;
  }
  t = nlohmann::json::parse(str);
  return true;
}
}  // namespace zenoh::ext::detail
