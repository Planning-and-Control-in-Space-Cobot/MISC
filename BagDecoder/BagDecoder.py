"""Copyright 2025  André Rebelo Teixeira.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import rosbag2_py
from rclpy.serialization import (
    deserialize_message,
)
from rosidl_runtime_py.utilities import (
    get_message,
)


class BagReader:
    def __init__(self, bag_path, topics, read_interval=None):
        """Initializes the BagReader class.

        Args:
            bag_path (str): Path to the rosbag file.
            topics (list): List of topics to read from the rosbag.
            read_interval (list, optional): A list with two integers representing the interval (in seconds) for reading messages.
        """
        self.bag_path = bag_path
        self.topics = topics
        self.topic_types = (
            {}
        )  # Dictionary to store topic types based on provided topics
        self.msgs = {}  # Dictionary to store the read messages per topic
        self.read_interval = None  # Initialize read interval
        self.bag_start_time = None
        self.bag_end_time = None

        # If a read_interval is provided, check if it's valid (two elements, and the first is smaller than the second).
        if read_interval is not None:
            if (len(read_interval) != 2) or (
                read_interval[0] > read_interval[1]
            ):
                # Warn if the interval is invalid and continue without applying it.
                print(
                    f"The read interval is not valid and it will be ignored, full bag will be read.\nThe Interval should be a list with two ints where the first one is smaller than the second one."
                )
            else:
                # Set the interval if valid.
                self.read_interval = read_interval

        # Create a reader for the rosbag file.
        reader = rosbag2_py.SequentialReader()

        # Open the bag file with the desired storage and serialization format.
        reader.open(
            rosbag2_py.StorageOptions(
                uri=bag_path,
                storage_id="mcap",  # Specify the bag format (e.g., mcap or sqlite3)
            ),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",  # Input and output serialization format
                output_serialization_format="cdr",
            ),
        )

        # Retrieve all topics and their types from the bag.
        topic_types = reader.get_all_topics_and_types()

        # Filter and store only the topics specified by the user.
        for topic in topic_types:
            if topic.name in self.topics:
                self.topic_types[topic.name] = get_message(
                    topic.type
                )  # Get the ROS message type for each topic
                self.msgs[topic.name] = (
                    []
                )  # Initialize an empty list for messages

        self.reader = reader  # Store the reader object

    def read_next_msg(
        self,
    ):
        """Reads the next message from the rosbag that matches the specified
        topics and read interval.

        Returns:
            tuple: Contains the message, its timestamp, and the topic. If no message is found, returns (None, None, None).
        """
        while self.reader.has_next():  # Loop until there are no more messages
            topic, data, timestamp = (
                self.reader.read_next()
            )  # Read the next message

            if self.bag_start_time is None:
                self.bag_start_time = timestamp

            self.bag_end_time = timestamp

            # Store the timestamp of the first message
            if not hasattr(self, "start_msg_timestamp"):
                self.start_msg_timestamp = timestamp

            # Check if the message should be read based on the timestamp and topic
            if self.check_read_msg(timestamp=timestamp, topic=topic):
                msg = deserialize_message(
                    data,
                    self.topic_types[topic],
                )
                return (
                    msg,  # The message itself
                    timestamp,  # The timestamp of the message
                    topic,  # The topic name
                )

        # Return None if no more messages are found
        return (
            None,
            None,
            None,
        )

    def read_bag(
        self,
    ):
        """Reads all messages from the rosbag that match the specified topics
        and interval.

        Stores the messages internally in a dictionary (self.msgs).
        """
        while True:
            msg, timestamp, topic = self.read_next_msg()

            # If the returned message or timestamp is None, stop reading
            if topic is None or msg is None or timestamp is None:
                break

            self.msgs[topic].append(
                (msg, timestamp)
            )  # Store the message along with its timestamp

    def get_msgs(
        self,
    ):
        """Returns all the messages read from the rosbag, grouped by topic.

        Returns:
            dict: A dictionary where the keys are topic names, and the values are lists of (message, timestamp) tuples.
        """
        return self.msgs

    def get_msg(
        self,
        topic,
    ):
        """Returns all the messages for a specific topic.

        Args:
            topic (str): The name of the topic to retrieve messages from.

        Returns:
            list: A list of tuples where each tuple contains a message and its corresponding float
        """
        return self.msgs[topic]

    def get_topics(
        self,
    ):
        """Returns the list of topics that were specified during initialization.

        Returns:
            list: A list of topic names.
        """
        return self.topics

    def get_topic_types(
        self,
    ):
        """Returns the types of the topics being read from the rosbag.

        Returns:
            dict: A dictionary where the keys are topic names and the values are message types.
        """
        return self.topic_types

    def get_read_interval(
        self,
    ):
        """Returns the read interval if it was set during initialization, or
        None if it was not set.

        Returns:
            list or None: A list with two integers representing the read interval, or None if no interval was set.
        """
        return self.read_interval

    def check_read_msg(self, timestamp: float, topic: str):
        """Checks if a message should be read based on the read interval and the
        topics specified in the constructor.

        Args:
            timestamp (float): Timestamp of the message.
            topic (str): Topic of the message.

        Returns:
            bool: True if the message should be read, False otherwise.
        """

        # Check if the message is inside the time interval and if the topic is valid
        if not self.check_msg_inside_interval(timestamp):
            return False

        if not topic in self.topics:
            return False

        return True

    def check_msg_inside_interval(self, timestamp: float) -> bool:
        """Checks if a message timestamp is inside the specified read interval.

        Args:
            timestamp (float): Timestamp to be checked.

        Returns:
            bool: True if the timestamp is inside the interval, False otherwise.
        """
        # If no interval was specified, read all messages
        if self.read_interval is None:
            return True

        first_message_time = self.timestamp_to_sec(self.start_msg_timestamp)

        # Check if the current message is within the specified interval relative to the first message
        if (
            self.timestamp_to_sec(timestamp)
            < first_message_time + self.read_interval[0]
            or self.timestamp_to_sec(timestamp)
            > first_message_time + self.read_interval[1]
        ):
            return False

        return True

    @staticmethod
    def timestamp_to_sec(timestamp: float):
        """Converts a float that comes in nanoseconds to seconds.

        Args:
            timestamp (float): Timestamp to be converted.

        Returns:
            float: The timestamp in seconds.
        """
        return timestamp / 1e9  # Convert nanoseconds to seconds

    @staticmethod
    def sec_to_timestamp(sec: float):
        """Converts a time in seconds to a float that comes in nanoseconds.

        Args:
            sec (float): Time in seconds.

        Returns:
            float: A timestamp in nanoseconds.
        """
        return float(sec * 1e9)  # Convert seconds to nanoseconds
