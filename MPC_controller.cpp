//
// Created by xushaohang
//
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <stdio.h>
#include "mpc_osqp.cc"
#include "fmpc/fmpc_cmd.h"
#include "fmpc/fmpc_status.h"
#include <iostream>
using namespace fmpc;

double _body_mass = 130/9.8;
std::vector<double> inertia = {0.17, 0, 0, 0, 0.57, 0, 0, 0, 0.64};
int num_legs = 4;
int planning_horizon = 10;
double timestep = 0.025;
std::vector<double> qp_weights = {1., 1., 0, 0, 0, 20, 0., 0., 0.1, 0.1, 0.1, 0.0, 0};
double alpha = 1e-5;
QPSolverName qp_solver_name = QPOASES;

ros::Publisher *fmpc_cmd_publisher;

ConvexMpc cpp_mpc=ConvexMpc(_body_mass,inertia,num_legs,planning_horizon,timestep,qp_weights,alpha,qp_solver_name);
void statusCallback(const fmpc_status::ConstPtr &msg)
{

        std::vector<double> contact_forces;
        std::vector<double> com_position(msg->com_position.cbegin(), msg->com_position.cend());
        std::vector<double> com_velocity_body_frame(msg->com_velocity_body_frame.cbegin(), msg->com_velocity_body_frame.cend());
        std::vector<double> com_roll_pitch_yaw(msg->com_roll_pitch_yaw.cbegin(), msg->com_roll_pitch_yaw.cend());
        std::vector<double> com_roll_pitch_yaw_rate(msg->com_roll_pitch_yaw_rate.cbegin(), msg->com_roll_pitch_yaw_rate.cend());
        std::vector<double> contact_estimates(msg->contact_estimates.cbegin(), msg->contact_estimates.cend());
        std::vector<double> foot_position_in_base_frame(msg->foot_position_in_base_frame.cbegin(), msg->foot_position_in_base_frame.cend());
        std::vector<double> friction_coeffs(msg->friction_coeffs.cbegin(), msg->friction_coeffs.cend());
        std::vector<double> desired_com_position(msg->desired_com_position.cbegin(), msg->desired_com_position.cend());
        std::vector<double> desired_com_velocity(msg->desired_com_velocity.cbegin(), msg->desired_com_velocity.cend());
        std::vector<double> desired_com_roll_pitch_yaw(msg->desired_com_roll_pitch_yaw.cbegin(), msg->desired_com_roll_pitch_yaw.cend());
        std::vector<double> desired_com_angular_velocity(msg->desired_com_angular_velocity.cbegin(), msg->desired_com_angular_velocity.cend());

        printf("  com_position   = (%f, %f, %f)\n", msg->com_position[0],msg->com_position[1],msg->com_position[2]);

        // compute contact_forces
       contact_forces = cpp_mpc.ComputeContactForces(    com_position,
                                                         com_velocity_body_frame,
                                                         com_roll_pitch_yaw,
                                                         com_roll_pitch_yaw_rate,
                                                         contact_estimates,
                                                         foot_position_in_base_frame,
                                                         friction_coeffs,
                                                         desired_com_position,
                                                         desired_com_velocity,
                                                         desired_com_roll_pitch_yaw,
                                                         desired_com_angular_velocity);
        fmpc_cmd cmd_msg;
        for (size_t i = 0; i < 12; i++)
        {
            cmd_msg.contact_force[i] = contact_forces[i];
        }
        fmpc_cmd_publisher->publish(cmd_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fmpc_node");
    ros::NodeHandle n;
    ros::Publisher fmpc_cmd_publisher_ = n.advertise<fmpc_cmd>("fmpc_cmd", 1000);
    fmpc_cmd_publisher = &fmpc_cmd_publisher_;
    ros::Subscriber sub = n.subscribe("fmpc_status", 1000, statusCallback);
    
    ros::spin();
    return 0;
}
