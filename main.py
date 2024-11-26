from lightning.pytorch.cli import LightningCLI
from modules.segdatamodule import SegDataModule
from modules.segmodule import SegModule


def main():
    LightningCLI(
        model_class=SegModule,
        datamodule_class=SegDataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == '__main__':
    main()
