cmake_minimum_required(VERSION 3.0.2)
project(fmpc)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)




add_message_files(
  FILES
  fmpc_cmd.msg
  fmpc_status.msg
)


generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)


catkin_package(

CATKIN_DEPENDS message_runtime
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  lib
  lib/qpoases/include
)
link_directories(lib)




add_executable(${PROJECT_NAME}_node MPC_controller.cpp)


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  libqpOASES.a
)



## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
SET(CMAKE_BUILD_TYPE Debug)
