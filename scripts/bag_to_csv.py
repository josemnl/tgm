import rosbag
from sensor_msgs import point_cloud2
import csv

log = '2024-03-15-11-56-01'
folder_path = './logs/' + log + '/'
bag_name = log + '.bag'
bag = rosbag.Bag(folder_path + bag_name)

# Print all topics
info = bag.get_type_and_topic_info()
for topic in info.topics:
	print(topic)

i = 0
for topic, msg, t in bag.read_messages(topics=['/ouster/points']):
	i = i+1
	with open(folder_path + 'z_' + str(i) + '.csv', 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile)
		for point in point_cloud2.read_points(msg, skip_nans=True):
			x = point[0]
			y = point[1]
			z = point[2]
			spamwriter.writerow([x, y, z])
bag.close()