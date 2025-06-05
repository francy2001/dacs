import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray as MsgFloat
from std_msgs.msg import Float64
from matplotlib.pyplot import get_cmap
import numpy as np
import traceback

class Visualizer(Node):
    def __init__(self):
        super().__init__(
            "visualizer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # [ use a timer ]
        time_period = 0.1 # second
        self.timer = self.create_timer(time_period, callback=self.timer_callback)

        # [ get parameters from launcher ]
        self.NN = self.get_parameter("N").value
        self.dd = self.get_parameter("d").value
        self.max_iter = self.get_parameter("max_iter").value
        _target_pos = self.get_parameter("target_pos").value
        self.target_pos = np.array(_target_pos).reshape((self.NN, self.dd))
        self.get_logger().info(f"target_pos: {self.target_pos}")

        # [ create subscriptions to all agents ]
        self.subscriptions_agent_state = []
        self.buffer_agent_state = { i : [] for i in range(self.NN) } # create FIFO buffers
        self.kk_states = 0
        
        self.subscriptions_agent_plot_info = []
        self.buffer_agent_plot_info = { i : [] for i in range(self.NN) } # create FIFO buffers
        self.kk_plot_info = 1

        self.targets_published = False
        self.cmap = get_cmap('Spectral')

        for i in range(self.NN):
            self.subscriptions_agent_state.append(
                self.create_subscription(
                    MsgFloat, 
                    f"/topic_agent_{i}_state",
                    self.listener_callback_state,
                    10
                )
            )

            self.subscriptions_agent_plot_info.append(
                self.create_subscription(
                    MsgFloat, 
                    f"/topic_agent_{i}_plot_info",
                    self.listener_callback_plot_info,
                    10
                )
            )

        # [ create the publishers that will communicate with Rviz ]
        self.publishers_agent = []
        for i in range(self.NN):
            self.publishers_agent.append(
                self.create_publisher(
                    Marker,
                    f"/visualization_topic_agent_{i}",
                    10
                )
            )
        
        self.publisher_cost = self.create_publisher(
            Float64,
            f"/visualization_topic_cost",
            10
        )
        self.publisher_norm_grad = self.create_publisher(
            Float64,
            f"/visualization_topic_norm_grad",
            10
        )
        self.publisher_barycenter = self.create_publisher(
            Marker,
            f"/visualization_topic_barycenter",
            10
        )
        
        self.publishers_target = []
        for i in range(self.NN):
            self.publishers_target.append(
                self.create_publisher(
                    Marker,
                    f"/visualization_target_{i}",
                    10
                )
            )


    def synchonized_data_in_queue(self, kk, queues):
        for i in range(self.NN):
            if len(queues[i]) == 0:
                # self.get_logger().info(f"Early exit. queue[{i}] was empty!")
                return False
        
        data_available = [(kk == queues[i][0][0]) for i in range(self.NN)]
        
        if not all(data_available):
            return False
        
        return True


    def timer_callback(self):
        data_available_plots = self.synchonized_data_in_queue(self.kk_plot_info, self.buffer_agent_plot_info)
        data_available_states = self.synchonized_data_in_queue(self.kk_states, self.buffer_agent_state)
        
        if not data_available_plots and not data_available_states:
            return

        if not self.targets_published:
            for i in range(self.NN):
                # [ publish target posisitions ]
                self.publish_marker_to_viz(self.publishers_target[i], self.target_pos[i], i, f"target_{i}", Marker.CYLINDER)
            self.targets_published = True


        if data_available_plots:
            def unmarshalling(msg):                    
                k = msg[0]
                cost = msg[1]
                grad = np.array(msg[2:])
                return k, cost, grad

            # [ extract N top messages ]
            total_cost = 0
            total_grad = np.zeros(shape=(self.dd))
            for i in range(self.NN):
                msg_i = self.buffer_agent_plot_info[i].pop(0)
                k, cost, grad = unmarshalling(msg_i)
                total_grad += grad
                total_cost += cost

            self.publish_plot_info_to_viz(total_cost, total_grad)
            self.kk_plot_info += 1

        if data_available_states:
            def unmarshalling(msg):
                k = msg[0]
                zz = msg[1            :  1+self.dd]   # 1st slice
                ss = msg[1+self.dd    :  1+2*self.dd] # 2nd slice
                vv = msg[1+2*self.dd  : ]             # 3rd slice
                return k, zz, ss, vv

            barycenter = np.zeros(shape=(self.dd))
            # [ extract N top messages ]
            for i in range(self.NN):
                msg_i = self.buffer_agent_state[i].pop(0)                
                k, zz_i, _, _ = unmarshalling(msg_i)
                barycenter += zz_i

                # [ publish agent state ]
                self.publish_marker_to_viz(self.publishers_agent[i], zz_i, i, f"agent_{i}")

            # [ publish barycenter ]
            barycenter = barycenter / self.NN
            self.publish_marker_to_viz(self.publisher_barycenter, barycenter, self.NN, f"barycenter", type=Marker.CUBE, scale=0.25)
            self.kk_states += 1

        # [ check max_iter limit ]
        self.get_logger().info(f"self.kk_plot_info: {self.kk_plot_info}")
        self.get_logger().info(f"self.kk_states: {self.kk_states}")
        if self.kk_plot_info >= self.max_iter and self.kk_states >= self.max_iter:
            self.get_logger().info(f"Maximum iteration reached. \n Bye!")
            raise SystemExit()
        

    def listener_callback_plot_info(self, msg):
        i = msg.data[0]
        msg_i = msg.data[1:]
        self.get_logger().info(f"listener_callback_plot_info: {i} {msg_i}")
        self.buffer_agent_plot_info[i].append(msg_i)

    def listener_callback_state(self, msg):
        i = int(msg.data[0])
        msg_i = msg.data[1:]
        self.get_logger().info(f"listener_callback_state: {i} {msg_i}")
        self.buffer_agent_state[i].append(msg_i)

    def publish_plot_info_to_viz(self, total_cost, total_grad):
        # [ publish info on the topic ]
        msg = Float64()
        msg.data = float(total_cost)
        self.get_logger().info(f"Publishing (/visualization_topic_cost): {total_cost}")
        self.publisher_cost.publish(msg)

        msg = Float64()
        total_grad_norm = np.linalg.norm(total_grad)
        msg.data = float(total_grad_norm)
        self.get_logger().info(f"Publishing (/visualization_topic_norm_grad): {total_grad_norm}")
        self.publisher_norm_grad.publish(msg)

    def publish_marker_to_viz(self, pub, pos, idx, ns, type=Marker.SPHERE, scale=0.75):
        # [ prepare the message ]
        marker = Marker()
            
        # [ set the pose of the marker ]
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]

        # [ specify the color of the marker as RGBA ]
        rgba = self.cmap(1/self.NN * idx)
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

        # [ other options ]
        marker.header.frame_id = "map"
        marker.ns = ns
        marker.id = idx
        marker.type = type
        marker.action = Marker.ADD
        marker.scale.x = 0.75
        marker.scale.y = 0.75
        marker.scale.z = 0.75

        # [ publish the message ]
        pub.publish(marker)



def main(args=None):
    """
    The main function.
    :param args: Not used directly by the user, but used by ROS2 to configure
    certain aspects of the Node.
    """
    rclpy.init(args=args)
    agent = Visualizer()
    try:
        agent.get_logger().info(f"Visualizer created")

        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("Terminated by KeyboardInterrupt")
    except SystemExit:
        agent.get_logger().info("Terminated by SystemExit")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
