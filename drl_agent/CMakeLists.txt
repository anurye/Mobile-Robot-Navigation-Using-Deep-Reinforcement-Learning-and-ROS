cmake_minimum_required(VERSION 3.8)
project(drl_agent)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(drl_agent_gazebo REQUIRED)
find_package(drl_agent_interfaces REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    launch
    scripts
    config
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/policies/td3_agent.py
    scripts/policies/train_td3_agent.py
    scripts/policies/test_td3_agent.py
    scripts/policies/td7_agent.py
    scripts/policies/train_td7_agent.py
    scripts/policies/test_td7_agent.py
    scripts/utils/replay_buffer.py
    scripts/utils/buffer.py
    scripts/utils/point_cloud2.py
    scripts/utils/load_yaml.py
    scripts/environment/environment_interface.py
    scripts/environment/environment.py
  DESTINATION 
    lib/${PROJECT_NAME}
)

ament_package()

