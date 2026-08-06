from qsrlib_io.world_trace import World_Trace, Object_State
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from collections import Counter

class QSRPipeline:

    def __init__(self, window_size=10):
        self.qsrlib = QSRlib()
        self.window_size = window_size

    # Entry Point

    def process_frames(self, collected_frames):

        if len(collected_frames) < self.window_size:
            return None

        frames = collected_frames[-self.window_size:]

        # Keeping only consistent objects across frames
        frames = self.filter_consistent_objects(frames)

        if len(frames) < 2:
            return None

        # Build world
        world = self.build_world_trace(frames)

        # Compute QTC
        response = self.compute_qtc(world)

        return response


    def filter_consistent_objects(self, frames): # Filtering consistent object in the frames
        # Keep only objects that appear in ALL frames

        object_sets = [
            set(obj["label"] for obj in frame["objects"])
            for frame in frames
        ]

        common_ids = set.intersection(*object_sets)

        if len(common_ids) < 2:
            return []

        filtered_frames = []

        for frame in frames:
            filtered_objects = [
                obj for obj in frame["objects"]
                if obj["label"] in common_ids
            ]

            filtered_frames.append({
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
                "objects": filtered_objects
            })

        return filtered_frames


    def build_world_trace(self, frames): # World Trace (Build)
        """
        Convert frames → QSRLib World Trace
        """

        world = World_Trace()
        object_tracks = {}

        for frame in frames:
            t = frame["timestamp"]  # use real time

            for obj in frame["objects"]:
                name = obj["label"]  # MUST be unique ID

                if name not in object_tracks:
                    object_tracks[name] = []

                object_tracks[name].append(
                    Object_State(
                        name=name,
                        timestamp=t,
                        x=obj["x"],
                        y=obj["z"] # use x z instead, height (y) is not as important
                    )
                )

        for obj_states in object_tracks.values():
            world.add_object_state_series(obj_states)

        return world



    def compute_qtc(self, world): # Computing QTC

        dynamic_args = {
            "qtcbs": {
                "quantisation_factor": 1.0,  # reduce noise
                "no_collapse": False
            }
        }

        request = QSRlib_Request_Message(
            which_qsr="qtcbs",
            input_data=world,
            dynamic_args=dynamic_args
        )

        return self.qsrlib.request_qsrs(request)

    # PRINT RESULTS
    def print_qtc(self, response):
        print("\n Scene analysis")

        # collect all person-object states across timesteps
        pair_states = {}

        for t, qsrs in response.qsrs.trace.items():
            for pair, relation in qsrs.qsrs.items():
                if 'person' not in pair:
                    continue # if no person we dont care
                if not pair.startswith('person'):
                    continue # only care about person to object relation
                if pair not in pair_states:
                    pair_states[pair] = []
                pair_states[pair].append(relation.qsr.get('qtcbs', ''))

        # print most common state for each person-object pair
        for pair, states in pair_states.items():
            most_common = Counter(states).most_common(1)[0][0]
            interpretation = self.interpret_qtc(most_common)
            print(f"  {pair}: {interpretation}")

    # INTERPRET QTC
    def interpret_qtc(self, qtc):
        """
        Simple behavior interpretation
        """

        if qtc == '+,+':
            return "approaching each other"
        elif qtc == '-,-':
            return "moving away from each other"
        elif qtc == '+,-':
            return "person moving toward object"
        elif qtc == '-,+':
            return "person moving away from object"
        elif qtc == '0,+':
            return "object moving toward person"
        elif qtc == '0,-':
            return "object moving away from person"
        elif qtc == '+,0':
            return "person approaching stationary object"
        elif qtc == '-,0':
            return "person leaving stationary object"
        elif qtc == '0,0':
            return "stationary"
        else:
            return "complex motion"