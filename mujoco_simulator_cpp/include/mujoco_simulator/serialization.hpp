#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <zenoh.hxx>

struct AttachModelRequst {
  std::string model_filename;
  std::string site_name;
  std::array<double, 3> pos;
  std::array<double, 4> quat;
  std::string prefix;
  std::string suffix;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AttachModelRequst, model_filename, site_name, pos, quat, prefix, suffix);

// inline bool __zenoh_serialize_with_serializer(zenoh::ext::Serializer& serializer,
//                                               const AttachModelRequst& s,
//                                               zenoh::ZResult* err) {
//   return zenoh::ext::detail::serialize_with_serializer(serializer, s.model_filename, err) &&
//          zenoh::ext::detail::serialize_with_serializer(serializer, s.site_name, err) &&
//          zenoh::ext::detail::serialize_with_serializer(serializer, s.pos, err) &&
//          zenoh::ext::detail::serialize_with_serializer(serializer, s.quat, err);
// }
//
// inline bool __zenoh_deserialize_with_deserializer(zenoh::ext::Deserializer& deserializer,
//                                                   AttachModelRequst& s,
//                                                   zenoh::ZResult* err) {
//   return zenoh::ext::detail::deserialize_with_deserializer(deserializer, s.model_filename, err) &&
//          zenoh::ext::detail::deserialize_with_deserializer(deserializer, s.site_name, err) &&
//          zenoh::ext::detail::deserialize_with_deserializer(deserializer, s.pos, err) &&
//          zenoh::ext::detail::deserialize_with_deserializer(deserializer, s.quat, err);
// }
