import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray as MsgFloat
import numpy as np
from time import sleep

# https://ros2-tutorial.readthedocs.io/en/latest/python_node_explained.html

class Agent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # [ extract configuration info ]
        self.agent_id = self.get_parameter("id").value
        self.neighbours = self.get_parameter("neighbours").value
        self.max_iter = self.get_parameter("max_iter").value
        
        _zz_init = self.get_parameter("zz_init").value
        self.zz_init = np.array(_zz_init)

        _adj_i = self.get_parameter("adj_i").value
        self.adj_i = np.array(_adj_i)

        self.dd = self.zz_init.shape[0]


        def sanity_check_args():
            self.get_logger().info(f"({type(self.agent_id)}) self.agent_id: {self.agent_id:d}")
            self.get_logger().info(f"({type(self.neighbours)}) self.neighbours: {self.neighbours}")
            self.get_logger().info(f"({type(self.zz_init)}) self.zz_init: {self.zz_init}")
            self.get_logger().info(f"({type(self.adj_i)}) self.adj_i: {self.adj_i}")
            self.get_logger().info(f"({type(self.dd)}) self.dd: {self.dd}")
            
        sanity_check_args()

        # [ create a publisher ]
        # you're changing the namespace, so you need a "/"
        self.publisher = self.create_publisher(
            MsgFloat, 
            f"/topic_agent_{self.agent_id}",
            10
        )
        
        # [ use a timer ]  // to check if I can consume/publish a message
        time_period = 1 # second
        self.timer = self.create_timer(time_period, callback=self.timer_callback)

        # [ create subscriptions ]
        self.my_subscriptions = []
        self.received_data = {j : [] for j in self.neighbours} # create FIFO buffers
        
        # TODO: NOTE: it's a bit strange to subscribe to own topic but it's more "uniform"
        # all the values and all the weights... including myself!
        for j in self.neighbours:
            self.my_subscriptions.append(
                self.create_subscription(
                    MsgFloat, 
                    f"/topic_agent_{j}",
                    # lambda msg: self.listener_callback(msg, j=j), # 2nd level function to pass also the source
                    self.listener_callback,
                    10
                )
            )
        
        self.kk = 0

    def aggregative_tracking(self, zz_neighbours):
        # [ my neighbour weighting row ]
        self.adj_i

        self.get_logger().info(f"[k:{self.kk}] Aggregative Tracking Update Step")
        
        cost = 0
        grad = np.zeros(shape=(self.dd))
        zz = np.zeros(shape=(self.dd))
        ss = np.zeros(shape=(self.dd))
        vv = np.zeros(shape=(self.dd))
        
        return (cost, grad, zz, ss, vv)

    def listener_callback(self, msg):
        self.get_logger().info(f"[k:{self.kk}] I heard: {msg.data}")
        j = int(msg.data[0])                # extract the index of the source
        msg_j = msg.data[1:]                # extract the remaining part of the message
                                            # in the 0 position there will be k (*NOTE*)
        self.received_data[j].append(msg_j) # push the received message on the buffer

        # self.get_logger().info(f"\t j: {j}")
        # self.get_logger().info(f"\t msg_j: {msg_j}")
        

    def timer_callback(self): # algorithm callback, do the local update
        # [ Message format ]
        # when publishing:
        #      - msg.data[0]   : agent_id
        #      - msg.data[1]   : k
        #      - msg.data[2:]  : state components, d=2
        # when consuming:      # NOTE: agent id have already been removed
        #      - msg.data[0]   : k
        #      - msg.data[1:]  : state components, d=2

        if self.kk == 0:
            self.zz_i = self.zz_init

            # TODO: https://roboticsbackend.com/ros2-create-custom-message/
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.kk), *list(self.zz_i)]
            
            self.publisher.publish(msg)
            self.get_logger().info(f"[k:{self.kk}] Publishing: {msg.data}")
            self.kk += 1


        else:
            # [ check if received data are synchronous at k-1 ]
            timestamp = [
                (self.kk - 1 == self.received_data[j][0][0]) for j in self.neighbours
            ]
            
            # [ if we are NOT synchronous, wait the next time callback ]
            if not all(timestamp):
                return
            
            # [ read the messages ]
            zz_neighbors = np.zeros(shape=(len(self.neighbours), self.dd))
            for idx, j in enumerate(self.neighbours):
                msg_j = self.received_data[j].pop(0)
                # self.get_logger().info(f"msg_{j}: {msg_j}")
                # kk_j = int(msg_j[0])
                zz_neighbors[idx] = np.array(msg_j[1:])

            self.get_logger().info(f"zz_neighbors: {zz_neighbors}")

            # [ compute the update ]
            #         zz_i_k_plus_1
            (cost, grad, zz, ss, vv) = self.aggregative_tracking(zz_neighbors)
            

            # [ publish on my own topic ]
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.kk), *list(self.zz)]
            self.publisher.publish(msg)
            self.get_logger().info(f"[k:{self.kk}] Publishing: {msg.data}")

            # [ publish to viz. ]
            # ...

            self.kk += 1

            # [ check max_iter limit ]
            if self.kk > self.max_iter:
                self.get_logger().info(f"[k:{self.kk}] Maximum iteration reached. \n Bye!")
                raise SystemExit()

def main(args=None):
    """
    The main function.
    :param args: Not used directly by the user, but used by ROS2 to configure
    certain aspects of the Node.
    """
    rclpy.init(args=args)
    agent = Agent()
    try:
        agent.get_logger().info(f"Agent ({agent.agent_id}) created")
        sleep(1)
        agent.get_logger().info("GO!")

        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("Terminated by KeyboardInterrupt")
    except SystemExit:
        agent.get_logger().info("Terminated by SystemExit")
    except Exception as e:
        print(e)
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
