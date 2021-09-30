import configparser

# .....................................................................................................................


class TbSETConfig:
    def __init__(self) -> None:
        """
        An object to expose the projects parameters and secrets.

        :return: None
        """

        config = configparser.ConfigParser()

        # CWD is tbset-project
        if len(config.read("tbset/tbset.ini")) > 0:
            self.TRN_HYPERP = {
                "num_layers": config["TRN_HYPERP"]["num_layers"],
                "d_model": config["TRN_HYPERP"]["d_model"],
                "num_heads": config["TRN_HYPERP"]["num_heads"],
                "dff": config["TRN_HYPERP"]["dff"],
                "dropout_rate": config["TRN_HYPERP"]["dropout_rate"],
                "ckpt_path": config["TRN_HYPERP"]["ckpt_path"],
                "save_path": config["TRN_HYPERP"]["save_path"],
                "epochs": config["TRN_HYPERP"]["epochs"]
            }
            self.DATASET_HYPERP = {
                "dwn_destination": config["DATASET_HYPERP"]["dwn_destination"],
                "vocab_path": config["DATASET_HYPERP"]["vocab_path"],
                "buffer_size": config["DATASET_HYPERP"]["buffer_size"],
                "batch_size": config["DATASET_HYPERP"]["batch_size"],
                "vocab_size": config["DATASET_HYPERP"]["vocab_size"],
                "num_examples": config["DATASET_HYPERP"]["num_examples"]
            }
        else:
            raise IOError("Impossible to open the config file for TbSET.")

    @staticmethod
    def update_config(section: str, field: str, value: str) -> None:
        """
        Updates "tbset/tbset.ini" file with provided values.

        :param section: The name of the configuration section where we want to make changes.
        :param field: The name of the configuration field we want to change.
        :param value: The new value we want to replace.

        :return: None
        """

        config = configparser.ConfigParser()
        config.read("tbset/tbset.ini")

        config.set(section, field, value)

        with open("tbset/tbset.ini", "w+") as config_file:
            config.write(config_file)

        config_file.close()
