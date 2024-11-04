// https://github.com/eclipse-zenoh/zenoh-cpp/tree/main/examples/universal
// https://github.com/eclipse-zenoh/zenoh-cpp/tree/main/examples/zenohc
// https://zenoh-cpp.readthedocs.io/en/stable/pubsub.html
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <iostream>

#include "zenoh.hxx"

using namespace zenoh;

int main(int argc, char** argv) {
  Config config = Config::create_default();
  auto session = Session::open(std::move(config));
  auto subscriber = session.declare_subscriber(
      KeyExpr("robot/qpos"),
      [](const Sample& sample) {
        spdlog::info("Received: {}", zenoh::ext::deserialize<std::vector<double>>(sample.get_payload()));
      },
      closures::none);
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
