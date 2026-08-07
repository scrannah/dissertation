# for ground truth world testing, llm and bayesian
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.expanduser("~/PycharmProjects/dissertation/src"))
sys.path.append(os.path.expanduser("~/PycharmProjects/dissertation/strands_qsr_lib/qsr_lib/src"))

from Agent import Agent
from Yolo_and_Conceptnet.conceptnet import get_info
import ollama

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import World_Trace, Object_State

TRACKED_OBJECTS = ["meal", "biscuits", "hobs", "plate", "sink", "glass", "bottle"]
CANDIDATE_GOALS = ["breakfast", "lunch", "drink", "unknown"]

QDC_PHRASES = {
    "touch": "is touching",
    "near": "is near",
    "medium": "is a medium distance from",
    "far": "is far from",
    "ignore": "is not relevant to"
}

QTC_PHRASES = {
    "-": "approaching",
    "+": "moving away from",
    "0": "stationary relative to"
}


def call_llm(prompt):
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "unclear"


class RobotAgentGroundTruth(Agent):
    def __init__(self):
        Agent.__init__(self, agentName="Tiago++")

        self.qsrlib = QSRlib()
        self.world_trace = World_Trace()

        qsrs_for = [("human", ooi) for ooi in TRACKED_OBJECTS] # only care about human-object relationships
        self.dynamic_args = {
            "argd": {
                "qsrs_for": qsrs_for,
                "qsr_relations_and_values": {"touch": 0.6, "near": 2, "medium": 3, "far": 5}
            },
            "qtcbs": {
                "qsrs_for": qsrs_for,
                "quantisation_factor": 0.01,
                "validate": False,
                "no_collapse": True
            },
            "mos": {
                "qsr_for": ["human"],
                "quantisation_factor": 0.09
            }
        }
        self.which_qsr = ["argd", "qtcbs", "mos"]

        self.conceptnet_cache = {}

        self.belief = {goal: 1.0 / len(CANDIDATE_GOALS) for goal in CANDIDATE_GOALS} # all goals as lilely as each  other at start
        self.confidence_threshold = 0.75

        self.timestep_counter = 0

        print(str(self.__class__.__name__) + " has activated (ground truth mode, qsrlib).")

    def perceive_ground_truth(self, debug=False): # debug for printing
        human_node = self.all_nodes.get("human")
        if human_node is None:
            return None

        hx, hy, hz = human_node.getPosition()

        held_field = human_node.getField("heldObjectReference") # gets if human is holding object
        held_id = held_field.getSFInt32() if held_field else 0
        holding_something = held_id != 0
        held_object_name = None
        if holding_something:
            held_node = self.supervisor.getFromId(held_id)
            if held_node is not None:
                held_object_name = held_node.getField("name").getSFString()

        self.timestep_counter += 1
        t = self.timestep_counter

        self.world_trace.add_object_state(Object_State(name="human", timestamp=t, x=hx, y=hy))

        for obj_name in TRACKED_OBJECTS:
            node = self.all_nodes.get(obj_name)
            if node is None:
                continue
            ox, oy, oz = node.getPosition()
            self.world_trace.add_object_state(Object_State(name=obj_name, timestamp=t, x=ox, y=oy))

        if self.timestep_counter < 2: # only computer if we have more than 2 timesteps
            if debug:
                print("Waiting for timestep")
            return None
        request_message = QSRlib_Request_Message(
            self.which_qsr, self.world_trace, dynamic_args=self.dynamic_args
        )
        response = self.qsrlib.request_qsrs(req_msg=request_message)

        relations = self.parse_qsr_response(response, held_object_name)

        if debug:
            print("Relations:")
            for r in relations:
                print(f"  human {QDC_PHRASES.get(r['qdc'], '?')} {r['object']}, "
                      f"{QTC_PHRASES.get(r['qtc'], '?')} it"
                      f"{' (HOLDING)' if r['holding'] else ''}")

        return {"relations": relations}

    def parse_qsr_response(self, response, held_object_name):
        relations = []

        timestamps = response.qsrs.get_sorted_timestamps()
        if not timestamps:
            return relations # in case empty list dont crash it
        latest_t = timestamps[-1] # only want most recent timestamp

        qsrs_at_t = response.qsrs.trace[latest_t].qsrs # grabs the qsr's from right now

        for pair_key, qsr_obj in qsrs_at_t.items():
            if not pair_key.startswith("human,"): # only care if human-object relationship
                continue
            obj_name = pair_key.split(",")[1] # just keep object name for prompting

            qsr_values = qsr_obj.qsr

            qdc_value = qsr_values.get("argd", "ignore") # if you cant find the qsr, default
            qtcbs_value = qsr_values.get("qtcbs", "0")
            qtc_value = qtcbs_value.split(",")[0] if qtcbs_value else "0" # only need first symbol for qtc, default to 0

            relations.append({
                "object": obj_name,
                "qdc": qdc_value,
                "qtc": qtc_value,
                "holding": (obj_name == held_object_name)
            })

        return relations

    def get_object_context(self, object_name):
        if object_name not in self.conceptnet_cache:
            self.conceptnet_cache[object_name] = get_info(object_name)
        return self.conceptnet_cache[object_name]

    def relations_to_text(self, relations):
        sentences = []
        for r in relations:
            if r["qdc"] == "ignore": # if object irrelevant do not build sentence
                continue

            sentence = (f"The human {QDC_PHRASES.get(r['qdc'], 'has an unknown relation to')} "
                        f"the {r['object']}, {QTC_PHRASES.get(r['qtc'], 'unknown movement')} it")
            # default to unknown ifd not found, sentence cannot be blank
            if r["holding"]:
                sentence += ", and is currently HOLDING it"
            sentence += "."

            concept_info = self.get_object_context(r["object"])
            if concept_info.get("used for"):
                sentence += f" ({r['object'].capitalize()} is typically used for {concept_info['used for'][0]}.)"
            elif concept_info.get("is a"):
                sentence += f" ({r['object'].capitalize()} is a type of {concept_info['is a'][0]}.)"

            sentences.append(sentence)

        if not sentences:
            return "The human is not near any known object." # if nothing to say give the llm something instead of empty
        return " ".join(sentences) # build sentence

    def compute_evidence_strength(self, relations):
        max_strength = 0.2
        for r in relations:
            if r["qdc"] == "ignore":
                continue
            strength = 0.2
            if r["qdc"] in ("touch", "near"):
                strength += 0.2
            if r["qtc"] == "-":
                strength += 0.2
            if r["holding"]:
                strength += 0.4
            max_strength = max(max_strength, strength)
        return min(max_strength, 1.0)

    def generate_hypothesis(self, observations, debug=True):
        relations = observations.get("relations", []) # grab your qsrs
        scene_description = self.relations_to_text(relations) # send them to sentences

        prompt = f"""You are observing a person in a kitchen. Based on the following
observation, what is the person most likely doing? Answer with a short
phrase only (e.g. "making breakfast", "cooking lunch", "getting a drink",
"unclear"). Do not explain your reasoning, just give the phrase.

Observation: {scene_description}

Answer:"""

        if debug:
            print("[LLM PROMPT]")
            print(prompt)

        hypothesis = call_llm(prompt)
        hypothesis = hypothesis.split("\n")[0].strip()

        if debug:
            print(f"[LLM HYPOTHESIS] {hypothesis}")

        return hypothesis

    def update_belief(self, hypothesis, evidence_strength=0.6, debug=True):
        matched_goal = "unknown"
        hypothesis_lower = hypothesis.lower()
        if "breakfast" in hypothesis_lower:
            matched_goal = "breakfast"
        elif "lunch" in hypothesis_lower or "cook" in hypothesis_lower:
            matched_goal = "lunch"
        elif "drink" in hypothesis_lower:
            matched_goal = "drink"

        for goal in self.belief:
            if goal == matched_goal:
                self.belief[goal] = self.belief[goal] + evidence_strength * (1 - self.belief[goal])
            else:
                self.belief[goal] = self.belief[goal] * (1 - evidence_strength * 0.3)

        total = sum(self.belief.values())
        for goal in self.belief:
            self.belief[goal] /= total

        if debug:
            print("[BELIEF]", {g: round(p, 3) for g, p in self.belief.items()})

        best_goal = max(self.belief, key=self.belief.get)
        best_confidence = self.belief[best_goal]

        if best_goal != "unknown" and best_confidence >= self.confidence_threshold:
            return best_goal, best_confidence

        return None

    def main(self):
        step_count = 0
        goal_locked = False

        while self.step():
            step_count += 1

            if step_count % 30 == 0 and not goal_locked:
                observations = self.perceive_ground_truth(debug=True)
                if observations and observations.get("relations"):
                    hypothesis = self.generate_hypothesis(observations, debug=True)
                    evidence_strength = self.compute_evidence_strength(observations["relations"])
                    result = self.update_belief(hypothesis, evidence_strength=evidence_strength, debug=True)
                    if result:
                        goal, confidence = result
                        print(f"Suggested goal: {goal} (confidence={confidence:.2f}) ")
                        goal_locked = True


def main():
    robot = RobotAgentGroundTruth()
    robot.main()


main()