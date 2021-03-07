from google.cloud import pubsub_v1
import os
import numpy
import json
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Munna/Documents/Github/RealityNavigation/auth/Sharath's Project-a05c51bd881f.json"
project_id='serene-athlete-271523'
topic_id = 'test_topic'
subscription_id = 'test_topic-sub'
timeout = 5.0
BUCKET='dataflow-eeg'
REGION='us-central1'


publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
topic_path = publisher.topic_path(project_id, topic_id)

for n in range(1, 10):
    data = "Message number " + str(n) + ": " + json.dumps(numpy.random.rand(30).tolist())
    # Data must be a bytestring
    data = data.encode("utf-8")
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data)
    print(future.result())

print(f"Published messages to {topic_path}.")


subscriber = pubsub_v1.SubscriberClient()
# The `subscription_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/subscriptions/{subscription_id}`
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def callback(message):
    print(f"Received {message}.")
    message.ack()

streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}..\n")

# Wrap subscriber in a 'with' block to automatically call close() when done.
with subscriber:
    try:
        # When `timeout` is not set, result() will block indefinitely,
        # unless an exception is encountered first.
        streaming_pull_future.result(timeout=timeout)
    except TimeoutError:
        streaming_pull_future.cancel()