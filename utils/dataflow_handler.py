import argparse
import datetime
import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam.transforms.window as window
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/LIINC/OneDrive/Documents/Sharath/RealityNavigation/auth/Sharath's Project-a05c51bd881f.json"
PROJECT='serene-athlete-271523'
BUCKET='dataflow-eeg'
REGION='us-central1'
topic_id = 'test_topic'

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


def run(input_topic, output_path, window_size=1.0):
    # `save_main_session` is set to true because some DoFn's rely on
    # globally imported modules.
    pipeline_options = PipelineOptions(
        runner='DataflowRunner',temp_location="gs://"+BUCKET+"/temp", project=PROJECT, region=REGION, streaming=True, save_main_session=True
    )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "Read PubSub Messages"
            >> beam.io.ReadFromPubSub(topic=input_topic)
            | "Window into" >> GroupWindowsIntoBatches(window_size)
            | "Write to GCS" >> beam.ParDo(WriteBatchesToGCS(output_path))
        )


if __name__ == "__main__":  # noqa
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--input_topic',
      dest='input_topic',
      default='projects/'+PROJECT+'/topics/'+topic_id,
      help="The Cloud Pub/Sub topic to read from.\n"
        '"projects/<PROJECT_NAME>/topics/<TOPIC_NAME>".')
    parser.add_argument(
      '--output_path',
      dest='output_path',
      default="gs://"+BUCKET+"/results/outputs",
      help="GCS Path of the output file including filename prefix.")
    parser.add_argument(
        "--window_size",
        type=float,
        default=1.0,
        help="Output file's window size in number of minutes.",
    )
    known_args, pipeline_args = parser.parse_known_args()

    run(
        known_args.input_topic,
        known_args.output_path,
        known_args.window_size,
    )