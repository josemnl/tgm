import rospy
import numpy as np
import time
# import occupancy_grid_map_py
from sensor_model.srv import SensorModelService, SensorModelServiceRequest
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray
from cgm_msgs.msg import ConvGridMap
import tf
from geometry_msgs.msg import TransformStamped

from tgm.sensorModel import sensorModel
from tgm.TGM import TGM
from tgm.SLAM import lsqnl_matching
from tgm.gridMap import gridMap
from tgm.lidarScan import lidarScan3D

class tgmNode:
    def __init__(self):

        self.latest_obstacle_point_cloud = None
        self.occupancyGrid_ = OccupancyGrid()

        self.obstacle_point_cloud_received = False
        self.ground_point_cloud_received = False
        self.occupancy_grid_received = False
        
        # Robot initial pose
        self.x_t = Float64MultiArray()
        self.x_t.data = [50, 50, 0] 

        # TGM parameters
        origin = [0,0]
        width = 400
        height = 400
        self.resolution = 2

        staticPrior = 0.3
        dynamicPrior = 0.3
        weatherPrior = 0
        # maxVelocity = 1/self.resolution
        maxVelocity = 1
        saturationLimits = [0, 1, 0, 1]
        fftConv = True

        # Sensor Model parameters
        self.smWidth = 100
        self.smHeight = 100
        self.sensorRange = 50
        invModel = [0.1, 0.9]
        occPrior = staticPrior + dynamicPrior + weatherPrior


        # Setup subscribers
        rospy.Subscriber("/obstacle_pointcloud", PointCloud2, self.obstacle_point_cloud_callback, queue_size=1)
        rospy.Subscriber("/ground_pointcloud", PointCloud2, self.ground_point_cloud_callback,queue_size=1)
        rospy.Subscriber("/grid_map", OccupancyGrid, self.occupancy_grid_callback,queue_size=1)
        self.static_grid_pub = rospy.Publisher("static_map", OccupancyGrid, queue_size=1)
        self.dynamic_grid_pub = rospy.Publisher("dynamic_map", OccupancyGrid, queue_size=1)
        self.grid_pub = rospy.Publisher("instantmap_custom", OccupancyGrid, queue_size=1)
        self.grid_pub2 = rospy.Publisher("instantmap_base", OccupancyGrid, queue_size=1)
        self.tgm_pub = rospy.Publisher("tgm_occupancy_grid",ConvGridMap, queue_size=1)
        self.br = tf.TransformBroadcaster()

        # Create Sensor Model and TGM
        self.sM = sensorModel(origin, self.smWidth, self.smHeight, self.resolution, self.sensorRange, invModel, occPrior)
        self.tgm = TGM(origin, width, height, self.resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv)
    
    def obstacle_point_cloud_callback(self, msg):
        self.latest_obstacle_point_cloud = msg
        self.obstacle_point_cloud_received = True
    
    def ground_point_cloud_callback(self, msg):
        self.latest_ground_point_cloud = msg
        self.ground_point_cloud_received = True
    
    # In case we generate the grid with the node
    def occupancy_grid_callback(self, msg):
        self.occupancyGrid_ = msg
        self.occupancy_grid_received= True
    
    def call_sensor_model(self, ground_pc, obstacle_pc, input_grid, x_t):
        
        rospy.wait_for_service('create_occupancy_grid',"")
        try:
            create_occupancy_grid = rospy.ServiceProxy('create_occupancy_grid', SensorModelService, True)
            req = SensorModelServiceRequest()
            req.ground_point_cloud = ground_pc
            req.obstacle_point_cloud = obstacle_pc
            req.input_grid = input_grid
            req.robot_pos = x_t
            res = create_occupancy_grid(req)
            
            return res.output_grid
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
    
    def initialize_grid(self):
        # Snap origin to align with the global grid's resolution
        origin_x = round((self.x_t.data[0] - self.smWidth / 2.0) * self.resolution) / self.resolution
        origin_y = round((self.x_t.data[1] - self.smHeight / 2.0) * self.resolution) / self.resolution
        
        self.occupancyGrid_.header.frame_id = "map"
        self.occupancyGrid_.header.stamp = rospy.Time.now()
        self.occupancyGrid_.info.resolution = 1 / self.resolution
        self.occupancyGrid_.info.width = self.smWidth * self.resolution
        self.occupancyGrid_.info.height = self.smHeight * self.resolution
        self.occupancyGrid_.info.origin.position.x = origin_x
        self.occupancyGrid_.info.origin.position.y = origin_y
        self.occupancyGrid_.info.origin.orientation.w = 1
        self.occupancyGrid_.data = [-1] * (self.occupancyGrid_.info.width * self.occupancyGrid_.info.height)  # Initialize the grid with UNKNOWN
    
    def convert_nav_msgs_to_gridMap(self):
         
        # Extract information from the nav_msgs/OccupancyGrid message
        width_number_cells = self.occupancyGrid_.info.width # Number of cells
        height_number_cells = self.occupancyGrid_.info.height  # Number of cells
        resolution_m_per_cell = self.occupancyGrid_.info.resolution  # Meters per cell
        origin = [self.occupancyGrid_.info.origin.position.x, self.occupancyGrid_.info.origin.position.y] 

        # Convert dimensions
        resolution_cells_per_meter = 1 / resolution_m_per_cell

        width_meters = int(width_number_cells * resolution_m_per_cell)
        height_meters = int(height_number_cells * resolution_m_per_cell)
        

        # Prepare data
        data = np.array(self.occupancyGrid_.data).reshape((height_number_cells, width_number_cells), order='F')

        # Custom mapping from original scale to new scale
        # UNKNOWN : 0.6, FREE_SPACE : 0.1, OBSTACLE : 0.9
        new_data = np.empty(data.shape, dtype=np.float32)
        new_data[data == -1] = 0.6 # Map UNKNOWN to 0.6
        new_data[data == 0] = 0.1   # Map FREE_SPACE to 0.1
        new_data[data == 100] = 0.9 # Map OBSTACLE to 0.9
        
        # Create gridMap instance
        
        grid_map = gridMap(origin, width_meters, height_meters, resolution_cells_per_meter, new_data)
        
        return grid_map
    
    def point_cloud2_to_lidarscan2D(self, msg):
        
        # Extract points as a list of tuples (x, y, z, ...)
        point_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Convert to a NumPy array (assuming each tuple has at least 3 elements for x, y, z)
        point_array = np.array(list(point_generator))[:, :3]  # Select only x, y, z if there are more fields
        
        z_t_3d = lidarScan3D(point_array)

        z_t = z_t_3d.convertTo2D().orderByAngle()

        return z_t
    
    def gridMap_to_convert_nav_msgs1(self, grid_map):
        #FUNCTION TO PRINT THE OCCUPANCY GRID GENERATED WITH THE INVERSE SENSOR MODEL
        
        grid_msg = OccupancyGrid()

        # Extract information from gridMap
        grid_msg.info.origin.position.x = grid_map.origin[0]
        grid_msg.info.origin.position.y = grid_map.origin[1]
        width_meters = grid_map.width
        height_meters = grid_map.height
        resolution_cells_per_meter = grid_map.resolution

        # Convert dimensions back
        resolution_m_per_cell = 1 / resolution_cells_per_meter
        width_cells = int(width_meters * resolution_cells_per_meter)
        height_cells = int(height_meters * resolution_cells_per_meter)
        grid_msg.info.resolution = resolution_m_per_cell 
        grid_msg.info.height = height_cells
        grid_msg.info.width = width_cells
        
        # Reverse the data mapping from the new scale to the original scale
        # 0.6 (UNKNOWN) to -1, 0.1 (FREE_SPACE) to 0, 0.9 (OBSTACLE) to 100
        original_data = np.array(grid_map.data, dtype=np.int8)
        original_data[grid_map.data == 0.6] = -1
        original_data[grid_map.data == 0.1] = 0
        original_data[grid_map.data == 0.9] = 100
        # grid_msg.data = original_data.flatten(order='F')
        grid_msg.data = original_data.flatten()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "base_link"
        
        self.grid_pub.publish(grid_msg)
    def gridMap_to_convert_nav_msgs2(self, grid_map):
        
        #FUNCTION TO PRINT THE OCCUPANCY GRID GENERATED WITH THE INVERSE SENSOR MODEL
        grid_msg = OccupancyGrid()
        # Extract information from gridMap
        
        grid_msg.info.origin.position.x = grid_map.origin[0]
        grid_msg.info.origin.position.y = grid_map.origin[1]
        width_meters = grid_map.width
        height_meters = grid_map.height
        resolution_cells_per_meter = grid_map.resolution

        # Convert dimensions back
        resolution_m_per_cell = 1 / resolution_cells_per_meter
        width_cells = int(width_meters * resolution_cells_per_meter)
        height_cells = int(height_meters * resolution_cells_per_meter)
        grid_msg.info.resolution = resolution_m_per_cell 
        grid_msg.info.height = height_cells
        grid_msg.info.width = width_cells
        
        
        # Reverse the data mapping from the new scale to the original scale
        # 0.6 (UNKNOWN) to -1, 0.1 (FREE_SPACE) to 0, 0.9 (OBSTACLE) to 100
        original_data = np.array(grid_map.data, dtype=np.int8)
        original_data[grid_map.data == 0.6] = -1
        original_data[grid_map.data == 0.1] = 0
        original_data[grid_map.data == 0.9] = 100
        # grid_msg.data = original_data.flatten(order='F')
        grid_msg.data = original_data.flatten()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map_ref"
        
        self.grid_pub2.publish(grid_msg)

    def publish_transform(self):
        # Extract position and orientation from x_t
        x, y, yaw = self.x_t.data
        self.br.sendTransform((x, y, 0),
                         tf.transformations.quaternion_from_euler(0, 0, yaw),
                         self.latest_obstacle_point_cloud.header.stamp,
                         #rospy.Time.now(),
                         "base_link",  # Child frame: /base_link
                         "map")        # Parent frame: /map
        
    def publish_tgm_map(self):
        m = ConvGridMap()
        m.header.frame_id = "map_ref"
        m.header.stamp = self.latest_obstacle_point_cloud.header.stamp
        m.info.width = self.tgm.staticMap.shape[1]
        m.info.height = self.tgm.staticMap.shape[0]
        m.info.resolution = 1.0 / self.tgm.resolution
        # Flatten the array and scale from 0...1 to 0...100
        flat_static_map = self.tgm.staticMap.ravel()
        flat_dynamic_map = self.tgm.dynamicMap.ravel()
        # In OccupancyGrid, -1 is unknown, 0 is free, 100 is occupied
        occupancy_grid_static = np.round(flat_static_map* 100).astype(int)
        occupancy_grid_dynamic = np.round(flat_dynamic_map* 100).astype(int)
        # Convert numpy array to list for the message
        m.static_data = list(occupancy_grid_static)
        m.dynamic_data = list(occupancy_grid_dynamic)
        self.tgm_pub.publish(m)
       
    
    def publish_static_map(self):
        
        # FUNCTION TO PRINT AND DEBUG THE STATIC MAP
        
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.latest_obstacle_point_cloud.header.stamp
        grid_msg.header.frame_id = "map"  # Change to your robot's reference frame as needed
        
        # Set the metadata
        grid_msg.info.resolution = 1.0 / self.tgm.resolution  # Grid resolution [meters/cell]

        grid_msg.info.width = self.tgm.staticMap.shape[1]  
        grid_msg.info.height = self.tgm.staticMap.shape[0]
        # Set the origin of the map 
        grid_msg.info.origin.position.x = self.tgm.origin[0]
        grid_msg.info.origin.position.y = self.tgm.origin[1]
        
        grid_msg.info.origin.orientation.x = 0.7071068
        grid_msg.info.origin.orientation.y = 0.7071068
        grid_msg.info.origin.orientation.z = 0
        grid_msg.info.origin.orientation.w = 0
        
        # Flatten the array and scale from 0...1 to 0...100
        flat_map = self.tgm.staticMap.ravel()
        # In OccupancyGrid, -1 is unknown, 0 is free, 100 is occupied
        occupancy_grid = np.round(flat_map * 100).astype(int)

        # Convert numpy array to list for the message
        grid_msg.data = list(occupancy_grid)

        # Publish the occupancy grid
        self.static_grid_pub.publish(grid_msg)
    
    def publish_dynamic_map(self):
        # FUNCTION TO PRINT AND DEBUG THE DYNAMIC MAP
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.latest_obstacle_point_cloud.header.stamp
        grid_msg.header.frame_id = "map"  # Change to your robot's reference frame as needed

        # Set the metadata
        grid_msg.info.resolution = 1.0 / self.tgm.resolution  # Grid resolution [meters/cell]

        grid_msg.info.width = self.tgm.dynamicMap.shape[1]  
        grid_msg.info.height = self.tgm.dynamicMap.shape[0]
        # Set the origin of the map
        grid_msg.info.origin.position.x = self.tgm.origin[0]
        grid_msg.info.origin.position.y = self.tgm.origin[1]

        grid_msg.info.origin.orientation.x = 0.7071068
        grid_msg.info.origin.orientation.y = 0.7071068
        grid_msg.info.origin.orientation.z = 0
        grid_msg.info.origin.orientation.w = 0
  
        # Flatten the array and scale from 0...1 to 0...100
        flat_map = self.tgm.dynamicMap.ravel()
        # In OccupancyGrid, -1 is unknown, 0 is free, 100 is occupied
        occupancy_grid = np.round(flat_map * 100).astype(int)

        # Convert numpy array to list for the message
        grid_msg.data = list(occupancy_grid)

        # Publish the occupancy grid
        self.dynamic_grid_pub.publish(grid_msg)
    
    # Main loop
    def process_data(self):     
        if self.obstacle_point_cloud_received and self.ground_point_cloud_received:    
            
            # Process the pointcloud
            
            # timeStart = time.time()
            z_t = self.point_cloud2_to_lidarscan2D(self.latest_obstacle_point_cloud)
            
            # timeData = time.time()
            
            # Execute SLAM
            # self.x_t.data = lsqnl_matching(z_t, self.tgm.computeStaticGridMap(), self.x_t.data, self.sensorRange).x
            
            # timeSLAM = time.time()
            
            # Publish the transform 
            self.publish_transform()
            
            # Compute instantaneous grid map with inverse sensor model
            
            ####### SM SERVICE ########
            self.sM.updateBasedOnPose(self.x_t.data)
            self.occupancyGrid_ = self.call_sensor_model(self.latest_ground_point_cloud, self.latest_obstacle_point_cloud, self.occupancyGrid_, self.x_t)
            gm = self.convert_nav_msgs_to_gridMap()
            self.gridMap_to_convert_nav_msgs1(gm)
            
            ####### SM BASE ########
            # self.sM.updateBasedOnPose(self.x_t.data)
            # gm = self.sM.generateGridMap(z_t, self.x_t.data)
            # self.gridMap_to_convert_nav_msgs2(gm)
           
            ####### SM NODE #######
            # gm = self.convert_nav_msgs_to_gridMap()
             
            # timeSensorModel = time.time()
            
            # Update TGM
            self.tgm.update(gm, self.x_t.data)
            
            # timeTGM = time.time()
            
            self.publish_tgm_map()
            # self.publish_static_map()
            # self.publish_dynamic_map()


            # Set the flags to False
            self.obstacle_point_cloud_received = False
            self.ground_point_cloud_received = False

            # Print times
            # print('Data:    ' + str(timeData - timeStart))
            # print('SLAM:    ' + str(timeSLAM - timeData))
            # print('InvSenM: ' + str(timeSensorModel - timeSLAM))
            # print('TGM:     ' + str(timeTGM - timeSensorModel))
            # print('Total:   ' + str(timeTGM  - timeStart))
            # print('')



    

def main():
    
    rospy.init_node('TGM_node', anonymous=True)
    print("nodo iniciado")
    tgm = tgmNode()
    
    rospy.Subscriber('/obstacle_pointcloud', PointCloud2, tgm.obstacle_point_cloud_callback)
    rospy.Subscriber('/ground_pointcloud', PointCloud2, tgm.ground_point_cloud_callback)
    rospy.Subscriber('/grid_map', OccupancyGrid, tgm.occupancy_grid_callback)
    
    tgm.initialize_grid()
    # Use a timer to periodically check and process data
    rospy.Timer(rospy.Duration(0.1), lambda event: tgm.process_data())

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
