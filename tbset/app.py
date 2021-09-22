import re
import tensorflow as tf

from prompt_toolkit import PromptSession
from tbset.utils.configuration import TbSETConfig
from tbset.src.translator import TrainingTranslator
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator, ValidationError, DummyValidator

# .....................................................................................................................


class TbSETValidator(Validator):

    def __init__(self, validation_type: str) -> None:
        """
        An object that implements several validation types for the user input.

        :param validation_type: A string to know which validation to apply
        :return: None
        """

        super().__init__()
        self.validation_type = validation_type

    def validate(self, document) -> None:
        """
        A method that raises an error when the typed "document" is invalid.

        :param document: A prompt_toolkit.document.Document object containing the user input
        :return: None
        """

        # document.text will have value in two cases, after we pressed enter in the prompt or when navigating down
        # the autocomplete commands list. In the second case there is no need to press enter to trigger this method,
        # but in those cases self.validation_type == ''
        typed = document.text

        if typed:
            if self.validation_type == "number":
                regex = r"^-?\d+$"

                if not re.search(regex, typed):

                    raise ValidationError(
                        message="Please input a positive or negative number."
                        )
            elif self.validation_type == "yes_no":
                regex = r"^[yYnN]$"

                if not re.search(regex, typed):
                    raise ValidationError(message="Please type y, n, Y or N.")
            elif self.validation_type == "text_max_len":
                if len(typed) > 100:
                    raise ValidationError(message="La oración debe tener menos de 100 caracteres.")
            else:
                raise ValidationError(message="Internal Error: Wrong validation type")


class TbSETPrompt:

    def __init__(self) -> None:
        """
        A shell object to interact with the translator.

        :return: None
        """

        self.config = TbSETConfig()
        self.session = PromptSession()
        self.commands = WordCompleter([
            "train",
            "translate"
        ])

    def _command_processor(self, cmd: str) -> None:
        """
        Handles the commands typed by the user.

        :param cmd: The command typed by the user.
        :return: None
        """

        if cmd == "translate":
            oracion = self.session.prompt("... Texto en español: ", validator=TbSETValidator("text_max_len"))
            self.translate(oracion)
        elif cmd == "train":
            confirmation = self.session.prompt("... This will take at least 30' with a GPU. Are you sure? (y/n): ",
                                               validator=TbSETValidator("yes_no"))

            if confirmation in "yY":
                self.train()
        else:
            print("Wrong command, please try again.\n")

    @staticmethod
    def _exit() -> None:
        """
        Allows exit the prompt.

        :return: None
        """

        print(
            "Thanks for using TbSET. "
            "See you next time!\n"
        )

    def start(self) -> None:
        """
        Infinite loop to handle user input in the TUI. It is needed to define a DummyValidator for the session prompt
        otherwise it will assign the selected dropdown item as document.text interfering with
        the TbSET_Validator.validate() method.

        :return: None
        """

        print("Bienvenido al traductor TbSET. Para salir presiona Ctrl-D.\n")

        while True:
            try:
                typed = self.session.prompt(
                    ".TbSET > ",
                    completer=self.commands,
                    validator=DummyValidator()
                    )
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            else:
                self._command_processor(typed)

        print("Bye!\n")

    def translate(self, oracion: str) -> None:
        """
        Translates the input text to English.

        :param oracion: A Spanish sentence. The length of the sentence should be less than or equal to 100 characters.
        :return: None
        """

        # Check if in the saved model path there is already a trained model
        saved_path = self.config.TRN_HYPERP["save_path"]

        if saved_path and tf.saved_model.contains_saved_model(saved_path):
            print("INFO: Trained model found. It will be used for inference.")
            # TODO: insert inference code here
        else:
            print("INFO: Couldn't find a saved model. Train the translator first with the `train` command.")

    def train(self) -> None:
        """
        Triggers the training proces of a Transformer model to translate Spanish sentences into English.

        :return: None
        """

        # Check if in the saved model path there is already a trained model
        if self.config.TRN_HYPERP["save_path"]:
            if tf.saved_model.contains_saved_model(self.config.TRN_HYPERP["save_path"]):
                print("INFO: An existing saved model will be used for inference")
            else:
                params = {**self.config.TRN_HYPERP, **self.config.DATASET_HYPERP}
                translator = TrainingTranslator(**params)
                translator.train()
                # TODO: Measure training time and inform results and how to use the trained model
        else:
            print("INFO: Path to save model wasn't provided in config file. Can't train the model")
