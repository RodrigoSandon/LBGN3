import pandas as pd
import os, glob
from typing import List
from pathlib import Path


def find_paths(root_path: Path, middle: str, endswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", middle, "**", endswith), recursive=True,
    )
    return files


class EventDatabase:
    def __init__(self) -> None:
        self.events: List[Event]
        self.events = []

    def add_event(self, event: str, event_path: Path) -> None:
        self.events.append(Event(event, event_path))


class Event:
    # example: "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/BLA-Insc-1/RDT D1/SingleCellAlignmentData/C01/Block_Reward Size_Choice Time (s)"
    def __init__(self, event: str, event_path: Path) -> None:
        self.event = event
        self.event_path = event_path

        self.mice: List[Mouse]
        self.mice = []

    def add_mouse(self, name, mouse_path) -> None:
        self.mice.append(Mouse(name, mouse_path))


class Mouse:
    def __init__(self, name: str, mouse_path: Path) -> None:
        self.name = name
        self.mouse_path = mouse_path


class TrialMatrix:
    def __init__(self, trial_number: int):
        pass


def main():

    """
    Goal: To have the decoder predict whether the mouse will choose Large, Small, or Omit based on before-choice activity.
    Training Data: Any data that will lead to a choice of L/S/O. Therefore includes these categorizations of data:
        1) Omission_Choice Time (s)
        2) Reward Size_Choice Time (s)

    But if we use these categorizations, we will be essentially repeating some data b/c the more generalized subcategorizations will include subcategorization of more
    specific subcategorizations. 

    The reason we may want to now include the data that are controlling -- wait nvm. It's better if we used the most generalized dataset b/c then we are 
    inputting inputs that can be of any trial type, whether shock ocurred or not, and any block (shock probability)
    """

    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis")

    files_1 = find_paths(
        ROOT_PATH, middle="Reward Size_Choice Time (s)", endswith="plot_ready.csv"
    )

    files_2 = find_paths(
        ROOT_PATH, middle="Omission_Choice Time (s)", endswith="plot_ready.csv"
    )

    # So now, if i get into one of these csv files under the same mouse, session, outcome, and same trial, I can start inputting them into a dict
    """
    Dict will look something like this:

        d = {
            [Outcome]: {
                [Mouse]: {
                    [Session]: {
                        [Trial]: {
                            [Cell df/f timewindow],
                            [Cell df/f timewindow],
                        }
                    },
                    [Session]: {
                        [Trial]: {
                            [Cell df/f timewindow],
                            [Cell df/f timewindow],
                        }
                    }
                },
                [Mouse]: {
                    [Session]: {
                        [Trial]: {
                            [Cell df/f timewindow],
                            [Cell df/f timewindow],
                        }
                    },
                    [Session]: {
                        [Trial]: {
                            [Cell df/f timewindow],
                            [Cell df/f timewindow],
                        }
                    }
                }
            },
    """


if __name__ == "__main__":
    main()
