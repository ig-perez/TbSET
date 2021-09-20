import re

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator, ValidationError, DummyValidator
from tbset.src.translator import Translator
from tbset.utils.configuration import TbSETConfig


class TbSETValidator(Validator):

    def __init__(self, validation_type: str) -> None:
        """
        An object that implements several validation for user input

        :param validation_type: A string to know which validation to apply
        """

        # Initializing our inherited class. If needed we can add parameters to
        # its __init__ method, if so, we instantiate it with the extra params
        super().__init__()

        self.validation_type = validation_type

    def validate(self, document) -> None:
        """
        A method that raises an error when the document is invalid

        :param document: A prompt_toolkit.document.Document object containing
                         the user input
        """

        # document.text will have value in two cases, after we pressed enter in
        # the prompt or when navigating down the autocomplete commands list. In
        # the second case there is no need to press enter to trigger this
        # method, but in those cases self.validation_type == ''
        typed = document.text

        if typed:
            if self.validation_type == "number":
                regex = r"^-?\d+$"

                if not re.search(regex, typed):

                    raise ValidationError(
                        message="Please input a positive or negative number"
                        )
            elif self.validation_type == "yes_no":
                regex = r"^[yYnN]$"

                if not re.search(regex, typed):
                    raise ValidationError(message="Please type y, n, Y or N")
            elif self.validation_type == "text_max_len":
                if len(typed) > 200:
                    raise ValidationError(message="La oración debe tener menos de 200 caracteres.")
            else:
                # We coded an invalid validation_type
                raise ValidationError(message="Internal Error: Wrong validation type")


class TbSETPrompt():
    """A shell object to interact with the translator"""

    def __init__(self) -> None:

        self.config = TbSETConfig()

        # An instance to handle inputs during the session (enables history)
        self.session = PromptSession()

        # Use a word completer to display available commands
        self.commands = WordCompleter([
            "train",
            "translate"
        ])

    def _command_processor(self, cmd: str) -> None:
        """
        Handles the commands typed by the user.

        :param cmd: The command typed by the user.
        """

        if cmd == "translate":
            oracion = self.session.prompt("... Texto en español: ", validator=TbSETValidator("text_max_len"))
            self.translate(oracion)
        elif cmd == "train":
            confirmation = self.session.prompt("... This will take at least 30' with a GPU. Are you sure? (y/n): ", validator=TbSETValidator("yes_no"))
            self.train()  # If previous input is not accepted, this line won't trigger
        else:
            print("Wrong command, please try again.\n")

    @staticmethod
    def _exit() -> None:
        """
        "To exit [bold yellow]Lucid[/bold yellow] press any of these: "
            "x, q, or Ctrl-D."
        """

        print(
            "Thanks for using TbSET. "
            "See you next time!\n"
        )

    def start(self) -> None:
        """
        Infinite loop to handle user input in the TUI.

        Note: It is needed to define a DummyValidator for the session prompt
        otherwise it will assign the selected dropdown item as document.text
        interfering with the TbSET_Validator.validate() method.
        """

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

    # .........................................................................

    def translate(self, oracion: str):
        print("in translate")

    def train(self):
        params = {**self.config.TRN_HYPERP, **self.config.DATASET_HYPERP}
        translator = Translator(**params)
        translator.train()
    # .........................................................................
