cmake_minimum_required(VERSION 3.0.2)
project(tgm)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
catkin_python_setup()

## Find catkin macros and libraries required for the build
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  pcl_conversions
  pcl_ros
  sensor_model  
  message_generation
  
)

find_package(Boost REQUIRED COMPONENTS python38)



###################################
## catkin specific configuration ##
###################################
generate_messages(
  DEPENDENCIES
  std_msgs  
  nav_msgs
)

catkin_package(
  # Since you're not defining a library in TGM, no LIBRARIES entry is needed here
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs pcl_conversions pcl_ros sensor_model
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)


#############
## Install ##
#############

## Mark Python scripts (nodes) for installation
catkin_install_python(PROGRAMS
  scripts/TGM_node.py  # Assuming the script is located in the 'scripts' directory
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)




## Mark other files for installation (e.g., launch and bag files, etc.)
install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
