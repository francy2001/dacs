import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray as MsgFloat
import numpy as np
from time import sleep
import traceback

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
        zz_init = np.array(_zz_init)

        _adj_i = self.get_parameter("adj_i").value
        self.adj_i = np.array(_adj_i)

        _target_pos = self.get_parameter("target_pos").value
        self.target_pos = np.array(_target_pos)

        self.gamma_1 = self.get_parameter("gamma_1").value
        self.gamma_2 = self.get_parameter("gamma_2").value

        self.alpha = self.get_parameter("alpha").value
        
        # needed for the 1/N coeffiecient in the \nabla_1 computation
        self.NN = self.get_parameter("N").value

        self.dd = zz_init.shape[0]

        def sanity_check_args():
            self.get_logger().info(f"({type(self.agent_id)}) self.agent_id: {self.agent_id:d}")
            self.get_logger().info(f"({type(self.neighbours)}) self.neighbours: {self.neighbours}")
            self.get_logger().info(f"({type(zz_init)}) zz_init: {zz_init}")
            self.get_logger().info(f"({type(self.adj_i)}) self.adj_i: {self.adj_i}")
            self.get_logger().info(f"({type(self.target_pos)}) self.dd: {self.target_pos}")
            self.get_logger().info(f"({type(self.gamma_1)}) self.gamma_1: {self.gamma_1}")
            self.get_logger().info(f"({type(self.gamma_2)}) self.gamma_2: {self.gamma_2}")
            self.get_logger().info(f"({type(self.NN)}) self.NN: {self.NN}")
            self.get_logger().info(f"({type(self.dd)}) self.dd: {self.dd}")
            
        # sanity_check_args()

        # [ create a publisher ]
        # you're changing the namespace, so you need a "/"
        self.publisher_state = self.create_publisher(
            MsgFloat, 
            f"/topic_agent_{self.agent_id}_state",
            10
        )
        self.publisher_plot_info = self.create_publisher(
            MsgFloat, 
            f"/topic_agent_{self.agent_id}_plot_info",
            10
        )
        
        # [ use a timer ]  // to check if I can consume/publish a message
        time_period = 0.1 # second
        self.timer = self.create_timer(time_period, callback=self.timer_callback)

        # [ create subscriptions ]
        self.my_subscriptions = []
        self.received_data = {j : [] for j in self.neighbours} # create FIFO buffers
        
        # NOTE: it's a bit strange to subscribe to own topic but it's more "uniform"
        # all the values and all the weights... including myself!
        for j in self.neighbours:
            self.my_subscriptions.append(
                self.create_subscription(
                    MsgFloat, 
                    f"/topic_agent_{j}_state",
                    # lambda msg: self.listener_callback(msg, j=j), # 2nd level function to pass also the source
                    self.listener_callback,
                    10
                )
            )
        
        self.kk = 0

        # [ three states ]
        # self.zz_i: positions of the agents
        # self.ss_i: proxy of \sigma(x^k)
        # self.vv_i: proxy of \frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_J(z_j^k, \sigma(z^k))

        # [ state initialization ]
        self.zz_i = zz_init
        self.ss_i = self.zz_i # \phi_{i}(z) is the identity function
        self.vv_i = self.gradient_computation(self.zz_i, self.ss_i, 'second')

    def cost_function(self):
        """
        Compute the cost function and its gradient for the aggregative tracking problem.
        """
        barycenter = self.ss_i # use the proxy
        target_norm = np.linalg.norm(self.zz_i - self.target_pos)
        barycenter_norm = np.linalg.norm(self.zz_i - barycenter)
        cost = self.gamma_1 * target_norm**2 + self.gamma_2 * barycenter_norm**2
        return cost


    def gradient_computation(self, zz_i, ss_i, type):
        # use the proxy
        barycenter = ss_i

        if type == 'first':
            # derivate of the cost function with respet to zz
            grad = 2 * self.gamma_1 * (zz_i - self.target_pos) + 2 * self.gamma_2 * (1 - 1/self.NN) * (zz_i - barycenter)
        elif type == 'second':
            # derivate of the cost function with respect to sigma(z)
            grad = -2 * self.gamma_2 * (zz_i - barycenter)
        else:
            raise ValueError("Invalid type. Use 'first' or 'second'.")
        return grad

    def aggregative_tracking(self, ss_neighbors, vv_neighbors):
        # self.get_logger().info(f"[k:{self.kk}] Aggregative Tracking Update Step")

        cost = self.cost_function()
        
        # [ zz update ]
        nabla_1 = self.gradient_computation(self.zz_i, self.ss_i, type='first')
        grad = nabla_1 + np.eye(self.dd) @ self.vv_i
        zz_i_k_plus_1 = self.zz_i - self.alpha * grad

        # [ ss update ]
        ss_consensus = ss_neighbors.T @ self.adj_i.T
        # ss_consensus = ss_consensus.squeeze()
        ss_local_innovation = zz_i_k_plus_1 - self.zz_i
        ss_i_k_plus_1 = ss_consensus + ss_local_innovation

        # [ vv update ]
        vv_consensus = vv_neighbors.T @ self.adj_i.T
        # vv_consensus = vv_consensus.squeeze()
        nabla2_k = self.gradient_computation(self.zz_i, self.ss_i, type='second')
        nabla2_k_plus_1 = self.gradient_computation(zz_i_k_plus_1, ss_i_k_plus_1, type='second')
        vv_local_innovation = nabla2_k_plus_1 - nabla2_k
        vv_i_k_plus_1 = vv_consensus + vv_local_innovation
        
        return cost, grad, zz_i_k_plus_1, ss_i_k_plus_1, vv_i_k_plus_1

    def listener_callback(self, msg):
        # self.get_logger().info(f"[k:{self.kk}] I heard: {msg.data}")
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
        #      - msg.data[2:]  : zz | ss | vv
        # when consuming:      # NOTE: agent id have already been removed
        #      - msg.data[0]   : k
        #      - msg.data[1:]  : zz | ss | vv

        if self.kk == 0:
            # TODO: https://roboticsbackend.com/ros2-create-custom-message/
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.kk), *list(self.zz_i), *list(self.ss_i), *list(self.vv_i)]

            self.publisher_state.publish(msg)
            # self.get_logger().info(f"[k:{self.kk}] Publishing (state)") # : {msg.data}")
            self.kk += 1

        else:
            for j in self.neighbours:
                if len(self.received_data[j]) == 0:
                    # self.get_logger().info(f"Early exit. self.received_data[{j}] was empty!")
                    return

            # [ check if received data are synchronous at k-1 ]
            data_available = [(self.kk - 1 == self.received_data[j][0][0]) for j in self.neighbours]
            
            # [ if we are NOT synchronous, wait the next time callback ]
            if not all(data_available):
                return
            
            # [ read the messages ]
            zz_neighbors = np.zeros(shape=(len(self.neighbours), self.dd))
            ss_neighbors = np.zeros(shape=(len(self.neighbours), self.dd))
            vv_neighbors = np.zeros(shape=(len(self.neighbours), self.dd))

            for idx, j in enumerate(self.neighbours):
                msg_j = self.received_data[j].pop(0)
                # self.get_logger().info(f"msg_{j}: {msg_j}")
                # kk_j = int(msg_j[0])
                zz_neighbors[idx] = np.array(msg_j[1            :  1+self.dd])   # 1st slice
                ss_neighbors[idx] = np.array(msg_j[1+self.dd    :  1+2*self.dd]) # 2nd slice
                vv_neighbors[idx] = np.array(msg_j[1+2*self.dd  : ])             # 3rd slice

            # self.get_logger().info(f"zz_neighbors: {zz_neighbors}")
            # self.get_logger().info(f"ss_neighbors: {ss_neighbors}")
            # self.get_logger().info(f"vv_neighbors: {vv_neighbors}")

            # [ compute the update ]
            #         zz_i_k_plus_1
            cost, grad, self.zz_i, self.ss_i, self.vv_i = self.aggregative_tracking(ss_neighbors, vv_neighbors)

            # [ publish on my own topic ]
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.kk), *list(self.zz_i), *list(self.ss_i), *list(self.vv_i)]
            # self.get_logger().info(f"[k:{self.kk}] Publishing (state)") #: {msg.data}")
            self.publisher_state.publish(msg)

            # [ publish to viz. ]
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.kk), float(cost), *list(grad)]
            # self.get_logger().info(f"[k:{self.kk}] Publishing (plot_info)") # : {msg.data}")
            self.publisher_plot_info.publish(msg)

            self.kk += 1

            # [ check max_iter limit ]
            if self.kk >= self.max_iter:
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
        agent.get_logger().info(f"Agent ({agent.agent_id}) created. GO!")
        sleep(5)
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
