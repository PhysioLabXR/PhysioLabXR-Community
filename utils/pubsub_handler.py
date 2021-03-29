from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import datetime
import json

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam.transforms.window as window
import os

class GroupWindowsIntoBatches(beam.PTransform):
    """A composite transform that groups Pub/Sub messages based on publish
    time and outputs a list of dictionaries, where each contains one message
    and its publish timestamp.
    """

    def __init__(self, window_size):
        # Convert minutes into seconds.
        self.window_size = int(window_size * 60)

    def expand(self, pcoll):
        return (
            pcoll
            # Assigns window info to each Pub/Sub message based on its
            # publish timestamp.
            | "Window into Fixed Intervals"
            >> beam.WindowInto(window.FixedWindows(self.window_size))
            | "Add timestamps to messages" >> beam.ParDo(AddTimestamps())
            # Use a dummy key to group the elements in the same window.
            # Note that all the elements in one window must fit into memory
            # for this. If the windowed elements do not fit into memory,
            # please consider using `beam.util.BatchElements`.
            # https://beam.apache.org/releases/pydoc/current/apache_beam.transforms.util.html#apache_beam.transforms.util.BatchElements
            | "Add Dummy Key" >> beam.Map(lambda elem: (None, elem))
            | "Groupby" >> beam.GroupByKey()
            | "Abandon Dummy Key" >> beam.MapTuple(lambda _, val: val)
        )


class AddTimestamps(beam.DoFn):
    def process(self, element, publish_time=beam.DoFn.TimestampParam):
        """Processes each incoming windowed element by extracting the Pub/Sub
        message and its publish timestamp into a dictionary. `publish_time`
        defaults to the publish timestamp returned by the Pub/Sub server. It
        is bound to each element by Beam at runtime.
        """

        yield {
            "message_body": element.decode("utf-8"),
            "publish_time": datetime.datetime.utcfromtimestamp(
                float(publish_time)
            ).strftime("%Y-%m-%d %H:%M:%S.%f"),
        }


class WriteBatchesToGCS(beam.DoFn):
    def __init__(self, output_path):
        self.output_path = output_path

    def process(self, batch, window=beam.DoFn.WindowParam):
        """Write one batch per file to a Google Cloud Storage bucket. """

        ts_format = "%H:%M"
        window_start = window.start.to_utc_datetime().strftime(ts_format)
        window_end = window.end.to_utc_datetime().strftime(ts_format)
        filename = "-".join([self.output_path, window_start, window_end])

        with beam.io.gcp.gcsio.GcsIO().open(filename=filename, mode="w") as f:
            for element in batch:
                f.write("{}\n".format(json.dumps(element)).encode("utf-8"))

class DataFlow:

    def __init__(self, credentials = "auth/Sharath's Project-a05c51bd881f.json", project_id = 'serene-athlete-271523',
                 bucket = 'dataflow-eeg', region='us-central1'):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(credentials)
        self.project_id = project_id
        self.topic_id = 'test_topic'
        self.subscription_id = 'test_topic-sub'
        self.timeout = 5.0
        self.BUCKET = bucket
        self.REGION = region
        self.count = 0
        self.output_path = "gs://"+self.BUCKET+"/results/outputs"
        self.pipeline_topic_id = 'projects/'+self.project_id+'/topics/'+self.topic_id

    def start_pipeline(self, window_size=1.0):
        # `save_main_session` is set to true because some DoFn's rely on
        # globally imported modules.
        pipeline_options = PipelineOptions(
            runner='DataflowRunner', temp_location="gs://" + self.BUCKET + "/temp", project=self.project_id, region=self.REGION,
            streaming=True, save_main_session=True
        )

        with beam.Pipeline(options=pipeline_options) as pipeline:
            (
                    pipeline
                    | "Read PubSub Messages"
                    >> beam.io.ReadFromPubSub(topic=self.pipeline_topic_id)
                    | "Window into" >> GroupWindowsIntoBatches(window_size)
                    | "Write to GCS" >> beam.ParDo(WriteBatchesToGCS(self.output_path))
            )

    def send_data(self, lsl_data_type,stream_data,timestamps):

        publisher = pubsub_v1.PublisherClient()
        # The `topic_path` method creates a fully qualified identifier
        # in the form `projects/{project_id}/topics/{topic_id}`
        topic_path = publisher.topic_path(self.project_id, self.topic_id)

        #if self.count < 10:
        print('timestamps',datetime.datetime.fromtimestamp(timestamps[0]))
        data = "Message number " + str(self.count) + "; Data type " + lsl_data_type + ": " + \
               json.dumps(stream_data.tolist()) + '; Timestamps: ' + json.dumps(timestamps.tolist())
        self.count += 1
        # Data must be a bytestring
        data = data.encode("utf-8")
        # When you publish a message, the client returns a future.
        future = publisher.publish(topic_path, data)