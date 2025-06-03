import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray as MsgFloat
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
        time_period = 1 # second
        self.timer = self.create_timer(time_period, callback=self.timer_callback)


        # [ get parameters from launcher ]
        self.NN = self.get_parameter("N").value
        self.dd = self.get_parameter("d").value

        self.kk = 0

        # [ create subscriptions to all agents ]
        self.subscriptions_agent_state = []
        self.buffer_agent_state = { i : [] for i in range(self.NN) } # create FIFO buffers
        
        self.subscriptions_agent_plot_info = []
        self.buffer_agent_plot_info = { i : [] for i in range(self.NN) } # create FIFO buffers

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
        self.publishers_vis = []
        for i in range(self.NN):
            self.publishers_vis.append(
                self.create_publisher(
                    Marker,
                    f"/visualization_topic_{i}",
                    10
                )
            )
        
        self.publisher_cost = self.create_publisher(
            Marker,
            f"/visualization_topic_cost",
            10
        )
        self.publisher_norm_grad = self.create_publisher(
            Marker,
            f"/visualization_topic_norm_grad",
            10
        )

        self.cmap = get_cmap('Spectral')

    def timer_callback(self):
        for i in range(self.NN):
            if len(self.buffer_agent_plot_info[i]) == 0:
                self.get_logger().info(f"Early exit. buffer_agent_plot_info[{i}] was empty!")
                return

        data_available = [(self.kk == self.buffer_agent_plot_info[i][0][0]) for i in range(self.NN)]
        
        # [ if we are NOT synchronous, wait the next time callback ]
        if not all(data_available):
            return

        # [ extract N top messages ]
        total_cost = 0
        total_grad = np.zero(shape=(self.dd))
        for i in range(self.NN):
            msg_i = self.received_data[i].pop(0)
            self.get_logger().info(f"msg_{i}: {msg_i}")
            k = msg_i[0]
            cost = msg_i[1]
            grad = np.array(msg_i[2:])
            
            total_grad += grad
            total_cost += cost

        # [ publish info on the topic ]
        msg = MsgFloat()
        msg.data = [float(self.kk), float(total_cost)]
        # self.get_logger().info(f"Publishing (/visualization_topic_cost):")
        self.publisher_cost.publish(msg)

        total_grad_norm = np.linalg.norm(total_grad)
        msg = MsgFloat()
        msg.data = [float(self.kk), float(total_grad_norm)]
        # self.get_logger().info(f"Publishing (/visualization_topic_norm_grad):")
        self.publisher_norm_grad.publish(msg)

        self.kk += 1
        

    def listener_callback_plot_info(self, msg):
        i = msg.data[0]
        k = msg.data[1]
        cost = msg.data[2]
        grad = msg.data[3:3+self.dd]

        msg_i = msg.data[1:]

        self.buffer_agent_plot_info[i].append(msg_i)

    def listener_callback_state(self, msg):
        # [ prepare the message ]
        marker = Marker()

        i = int(msg.data[0])
        k = msg.data[1]
        zz = msg.data[1            :  1+self.dd]   # 1st slice
        ss = msg.data[1+self.dd    :  1+2*self.dd] # 2nd slice
        vv = msg.data[1+2*self.dd  : ]             # 3rd slice    
            
        # [ set the pose of the marker ]
        marker.pose.position.x = zz[0]
        marker.pose.position.y = zz[1]

        # [ specify the color of the marker as RGBA ]
        rgba = self.cmap(1/self.NN * i)
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

        # [ other options ]
        marker.header.frame_id = "map"
        marker.ns = f"agent_{i}"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # [ publish the message ]
        self.get_logger().info(f"Publishing (/visualization_topic_{i}): {marker.pose.position}")
        self.publishers_vis[i].publish(marker)


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
