# for ground truth world testing, llm and bayesian
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.expanduser("~/PycharmProjects/dissertation/src"))
sys.path.append(os.path.expanduser("~/PycharmProjects/dissertation/strands_qsr_lib/qsr_lib/src"))

from Agent import Agent
from Yolo_and_Conceptnet.conceptnet import get_info
import ollama

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import World_Trace, Object_State

TRACKED_OBJECTS = ["meal", "biscuits", "hobs", "plate", "sink", "glass", "bottle"]
candidate_goals = ["breakfast", "lunch", "drink"]

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

        self.min_actions_for_goal_reasoning = 5
        self.lock_streak = 5
        self.consecutive_goal = None
        self.consecutive_count = 0
        self.action_history = []
        self.action_window = 5   # only feed the last 5 actions into stage 2, not the whole accumulated list
        self.goal_guess_history = []
        self.goal_window = 5
        self.min_guesses_for_likelihood = 3 # dont trust the frequency count until we have enough guesses banked
        self.belief = {goal: 1.0 / len(candidate_goals) for goal in candidate_goals} # all goals as likely as each  other at start
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

    def relations_to_text(self, relations, object_evidence_strength):
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
            sentence += f" (evidence strength: {object_evidence_strength[r['object']]})"
            if concept_info.get("used for"):
                sentence += f" {r['object'].capitalize()} is typically used for {concept_info['used for'][0]}."
            elif concept_info.get("is a"):
                sentence += f" {r['object'].capitalize()} is a type of {concept_info['is a'][0]}."

            sentences.append(sentence)

        if not sentences:
            return "The human is not near any known object." # if nothing to say give the llm something instead of empty
        return " ".join(sentences) # build sentence

    def compute_evidence_strength(self, relations):
        object_evidence_strengths = {}
        for r in relations:
            if r["qdc"] == "ignore":
                continue
            strength = 0.2 # reset strength per object
            if r["qdc"] in ("touch", "near"):
                strength += 0.2
            if r["qtc"] == "-":
                strength += 0.2
            if r["holding"]:
                strength += 0.4
            object_evidence_strengths[r["object"]] = strength # append strengths per obejct to dict
        return object_evidence_strengths

    def interpret_action(self, relations, object_evidence_strengths, debug=True):
        scene_description = self.relations_to_text(relations, object_evidence_strengths)

        prompt = f"""You are the perception stage in a multi-stage intention-recognition
pipeline. Errors here become facts to a later reasoning stage, so accuracy
matters more than completeness. Do not explain your reasoning.

In one short discrete phrase (like "taking the bottle to the glass" or
"using the plate at the hobs"), describe the action the person is
performing. Never invent a holding status that isn't explicitly stated
in the observation.

Example 1:
Observation: The human is touching the bottle, approaching it, and is
currently HOLDING it. The human is near the glass, approaching it.
Action: Taking the bottle to the glass.

Example 2:
Observation: The human is touching the meal, approaching it, and is
currently HOLDING it. The human is a medium distance from the hobs,
approaching it.
Action: Carrying the meal to the hobs.

Example 3:
Observation: The human is touching the plate, approaching it, and is
currently HOLDING it. The human is touching the hobs, stationary
relative to it.
Action: Using the plate at the hobs.

Example 4:
Observation: The human is near the plate, approaching it. The human is
far from the glass, stationary relative to it.
Action: Approaching the plate.

Example 5:
Observation: The human is touching the biscuits, stationary relative to
it, and is currently HOLDING it. The human is far from the sink, moving
away from it.
Action: Picking up the biscuits and moving away from the sink.

Example 6:
Observation: The human is far from the meal, stationary relative to it.
The human is far from the hobs, stationary relative to it.
Action: Not interacting with anything nearby.

Now here is the real situation.

Observation: {scene_description}

Action:"""

        if debug:
            print("[STAGE 1 PROMPT]")
            print(prompt)

        action = call_llm(prompt)
        action = action.strip()

        if debug:
            print(f"[STAGE 1 ACTION] {action}")

        return action

    def generate_hypothesis(self, action_history, debug=True):
        # no longer asks for JSON likelihoods, asks for a single vote
        # llm own likelihood value not grounded anything
        # coll;ect likelihood from banked votes
        history_text = "\n".join(f"- {a}" for a in action_history)

        prompt = f"""You are one vote in a Bayesian intention-recognition system. Your
single-word answer is combined with several other votes over time to build
confidence in a goal, repeated votes for the same goal push confidence up
quickly, so voting confidently on weak or repetitive evidence causes the
system to commit to a goal too early and too often. Because of this, you
should only vote for breakfast, lunch, or drink when the evidence genuinely
discriminates between them, being unsure is okay. Based on the following recent actions, which ONE of these is most
likely the person's goal: {', '.join(candidate_goals)}?

If the evidence does not clearly point to one goal, respond with "unsure"
instead. Respond with EXACTLY ONE WORD, one of {', '.join(candidate_goals)},
or "unsure" and nothing else.

Example 1:
Actions:
- The person is holding the bottle and moving toward the glass.
- The person is holding the glass, stationary near the sink.
Answer: drink

Example 2:
Actions:
- The person is holding the meal and moving toward the hobs.
- The person is holding the plate while standing at the hobs.
Answer: lunch

Example 3:
Actions:
- The person is holding the biscuits and moving away from the sink.
- The person is holding the meal and moving toward the plate.
Answer: breakfast

Example 4:
Actions:
- The person is not holding anything and is approaching the plate.
- The person is not holding anything and is approaching the glass.
Answer: unsure

Example 5:
Actions:
- The person is holding the meal and moving toward the hobs.
- The person is touching the hobs while holding the meal.
Answer: lunch

Example 6:
Actions:
- The person is not holding anything and is not moving toward any object.
- The person is not holding anything and is approaching the biscuits.
Answer: unsure


Now here is the real situation.

Recent actions observed, oldest first:
{history_text}

Answer:"""

        if debug:
            print("[LLM PROMPT]")
            print(prompt)

        raw_guess = call_llm(prompt).strip().lower()

        if debug:
            print(f"[LLM GUESS] {raw_guess}")

        if raw_guess not in candidate_goals: # unusable output, skip this timestep
            return None

        return raw_guess

    def compute_likelihoods_from_guesses(self): # try making bayesian from past guesses
        recent = self.goal_guess_history[-self.goal_window:] # this method only works with fixed goal list
        total = len(recent)
        return {goal: recent.count(goal) / total for goal in candidate_goals}

    def update_belief(self, likelihoods, debug=True):
        if likelihoods is None:
            return None # escape


        for goal in self.belief:
            self.belief[goal] = self.belief[goal] * max(likelihoods.get(goal, 1.0), 0.05)
            # default to 1 keeps bayesian goal update at same number, 0.05 floors it

        total = sum(self.belief.values())

        if total == 0:
            return None # escape if all values 0.0, breaks when divising for normalising

        for goal in self.belief:
            self.belief[goal] /= total

        if debug:
            print("[BELIEF]", {g: round(p, 3) for g, p in self.belief.items()})

        best_goal = max(self.belief, key=self.belief.get) # key makes it so max compares numbers not goals
        best_confidence = self.belief[best_goal]

        if best_confidence >= self.confidence_threshold:
            if best_goal == self.consecutive_goal:
                self.consecutive_count += 1
            else:
                self.consecutive_goal = best_goal
                self.consecutive_count = 1

            if self.consecutive_count >= self.lock_streak:
                return best_goal, best_confidence

        return None

    def main(self):
        step_count = 0
        goal_locked = False

        while self.step():
            step_count += 1

            if step_count % 90 == 0 and not goal_locked: # fine tune  for more/less movements caught
                observations = self.perceive_ground_truth(debug=True)
                if observations and observations.get("relations"):
                    object_evidence_strengths = self.compute_evidence_strength(observations.get("relations"))
                    action = self.interpret_action(observations.get("relations"), object_evidence_strengths)
                    if not self.action_history or self.action_history[-1] != action:
                        self.action_history.append(action)
                        if len(self.action_history) > self.action_window:
                            self.action_history.pop(0)
                    if len(self.action_history) >= self.min_actions_for_goal_reasoning:
                        guess = self.generate_hypothesis(self.action_history, debug=True)
                        if guess is not None:
                            self.goal_guess_history.append(guess)
                            if len(self.goal_guess_history) >= self.min_guesses_for_likelihood: # dont trust a 1 guess frequency
                                likelihoods = self.compute_likelihoods_from_guesses() # likelihood
                                result = self.update_belief(likelihoods, debug=True)
                                if result:
                                    goal, confidence = result
                                    print(f"Suggested goal: {goal} (confidence={confidence:.2f}) ")
                                    goal_locked = True # maybe dont lock when only one goal is certain?
                                    # TODO once goal locked using past observations create action plan?


def main():
    robot = RobotAgentGroundTruth()
    robot.main()


main()