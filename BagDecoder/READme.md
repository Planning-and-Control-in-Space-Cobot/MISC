# BagReader

## Overview
The `BagReader` class provides an interface to read messages from a ROS2 rosbag (MCAP format). It allows users to filter messages by topic and optionally specify a time interval for reading messages.

## Features
- Reads messages from a ROS2 rosbag.
- Filters messages by topic.
- Supports optional time interval for selective reading.
- Stores messages with timestamps for later retrieval.

## Dependencies
This script requires:
- `rosbag2_py`
- `rclpy.serialization`
- `rosidl_runtime_py.utilities`

Ensure that ROS2 is properly installed in your system and the necessary dependencies are available.

## Installation
To use this script, ensure you have a ROS2 environment set up and activate it before running the script:
```sh
source /opt/ros/humble/setup.bash  # Adjust based on your ROS2 distribution
```

## Usage
To use the `BagReader` class, initialize it with the path to a ROS2 bag file and the list of topics to read:

```python
from bag_reader import BagReader

bag_path = "path/to/your/rosbag"
topics = ["/topic1", "/topic2"]
read_interval = [5, 15]  # Optional time interval in seconds

reader = BagReader(bag_path, topics, read_interval)
reader.read_bag()

messages = reader.get_msgs()
print(messages)
```

## Methods
- `read_next_msg()`: Reads the next message from the bag.
- `read_bag()`: Reads all messages from the specified topics.
- `get_msgs()`: Retrieves all stored messages.
- `get_msg(topic)`: Retrieves messages for a specific topic.
- `get_topics()`: Returns the list of topics being read.
- `get_topic_types()`: Returns the topic types.
- `get_read_interval()`: Returns the read interval if specified.
- `check_read_msg(timestamp, topic)`: Checks if a message should be read.
- `check_msg_inside_interval(timestamp)`: Checks if a message timestamp falls within the specified interval.
- `timestamp_to_sec(timestamp)`: Converts a nanosecond timestamp to seconds.
- `sec_to_timestamp(sec)`: Converts seconds to a nanosecond timestamp.

## Notes
- The script assumes the rosbag is in MCAP format (`storage_id="mcap"`). Modify accordingly if using a different format.
- If the specified read interval is invalid, the script will read the full bag without filtering by time.

## License
This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Author
André Rebelo Teixeira

