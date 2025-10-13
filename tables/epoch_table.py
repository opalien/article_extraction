# Model	Domain	Task	Organization	Authors	Publication date	Reference	Link	Citations	Notability criteria	Notability criteria notes	Parameters	Parameters notes	Training compute (FLOP)	Training compute notes	Training dataset	Training dataset notes	Training dataset size (datapoints)	Dataset size notes	Training time (hours)	Training time notes	Training hardware	Approach	Confidence	Abstract	Epochs	Benchmark data	Model accessibility	Country (of organization)	Base model	Finetune compute (FLOP)	Finetune compute notes	Hardware quantity	Hardware utilization (MFU)	Last modified	Training cloud compute vendor	Training data center	Archived links	Batch size	Batch size notes	Organization categorization	Foundation model	Training compute lower bound	Training compute upper bound	Training chip-hours	Training code accessibility	Accessibility notes	Organization categorization (from Organization)	Possibly over 1e23 FLOP	Training compute cost (2023 USD)	Utilization notes	Numerical format	Frontier model	Training power draw (W)	Training compute estimation method	Hugging Face developer id	Post-training compute (FLOP)	Post-training compute notes	Hardware utilization (HFU)
# Grok 4 Fast	Language	Language modeling/generation,Question answering,Quantitative reasoning,Search,Code generation,Mathematical reasoning	xAI		2025-09-19	Pushing the Frontier of Cost-Efficient Intelligence	https://x.ai/news/grok-4-fast								Unspecified unreleased	"Grok 4 Fast is first pre-trained with a data recipe that includes publicly available Internet data, data produced by third-parties for xAI, data from users or contractors, and internally generated data. We perform data filtering procedures on the training data, such as de-duplication and classification, to ensure data quality and safety prior to training. In addition to pre-training, our recipe uses a variety of reinforcement learning techniques—human feedback, verifiable rewards, and model grading—along with supervised finetuning of specific capabilities."							Unknown	We're thrilled to present Grok 4 Fast, our latest advancement in cost-efficient reasoning models. Built on xAI’s learnings from Grok 4, Grok 4 Fast delivers frontier-level performance across Enterprise and Consumer domains—with exceptional token efficiency. This model pushes the boundaries for smaller and faster AI, making high-quality reasoning accessible to more users and developers. Grok 4 Fast features state-of-the-art (SOTA) cost-efficiency, cutting-edge web and X search capabilities, a 2M token context window, and a unified architecture that blends reasoning and non-reasoning modes in one model.			API access	United States of America						2025-09-22 14:40:04+00:00						Industry					Unreleased		Industry


from __future__ import annotations
from datetime import date, datetime
from typing import Any, Type, TypeVar, cast
from types import TracebackType
from pathlib import Path
import pandas as pd  # type: ignore[reportMissingTypeStubs]
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table
from sqlalchemy.types import Integer, String, Date, Float, Text, DateTime, Boolean

from .base import Base
from .other.utils import download_zip, extract_zip
from .other.columns import (
    COLUMN_MAPPING,
    COLUMN_ORDER,
    COLUMN_CSV_TO_ATTR,
    DATE_COLUMNS,
    DATETIME_COLUMNS,
    INTEGER_COLUMNS,
    FLOAT_COLUMNS,
    BOOLEAN_COLUMNS,
)
from .other.convert import to_date, to_datetime, to_boolean, to_float, to_int
from .other.split import normalize_epoch_dataframe, shuffled_split

EPOCH_TABLE_URL = "http://epoch.ai/data/generated/ai_models.zip"
CACHE_DIR = Path(".cache/epoch_table")

class nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None


class EpochColumns:
    __abstract__ = True

    with nothing():
        id_paper: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        model: Mapped[str | None] = mapped_column(String, nullable=True)
        domain: Mapped[str | None] = mapped_column(String, nullable=True)
        task: Mapped[str | None] = mapped_column(String, nullable=True)
        organization: Mapped[str | None] = mapped_column(String, nullable=True)
        authors: Mapped[str | None] = mapped_column(String, nullable=True)
        publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
        reference: Mapped[str | None] = mapped_column(String, nullable=True)
        link: Mapped[str | None] = mapped_column(String, nullable=True)
        citations: Mapped[int | None] = mapped_column(Integer, nullable=True)
        notability_criteria: Mapped[str | None] = mapped_column(String, nullable=True)
        notability_criteria_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        parameters: Mapped[float | None] = mapped_column(Float, nullable=True)
        parameters_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        training_compute: Mapped[float | None] = mapped_column(Float, nullable=True)
        training_compute_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        training_dataset: Mapped[str | None] = mapped_column(String, nullable=True)
        training_dataset_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        training_dataset_size_datapoints: Mapped[float | None] = mapped_column(Float, nullable=True)
        dataset_size_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        training_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
        training_time_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        training_hardware: Mapped[str | None] = mapped_column(String, nullable=True)
        approach: Mapped[str | None] = mapped_column(String, nullable=True)
        confidence: Mapped[str | None] = mapped_column(String, nullable=True)
        abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
        epochs: Mapped[float | None] = mapped_column(Float, nullable=True)
        benchmark_data: Mapped[str | None] = mapped_column(String, nullable=True)
        model_accessibility: Mapped[str | None] = mapped_column(String, nullable=True)
        country_of_organization: Mapped[str | None] = mapped_column(String, nullable=True)
        base_model: Mapped[str | None] = mapped_column(String, nullable=True)
        finetune_compute: Mapped[float | None] = mapped_column(Float, nullable=True)
        finetune_compute_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        hardware_quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
        hardware_utilization_mfu: Mapped[float | None] = mapped_column(Float, nullable=True)
        last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
        training_cloud_compute_vendor: Mapped[str | None] = mapped_column(String, nullable=True)
        training_data_center: Mapped[str | None] = mapped_column(String, nullable=True)
        archived_links: Mapped[str | None] = mapped_column(String, nullable=True)
        batch_size: Mapped[float | None] = mapped_column(Float, nullable=True)
        batch_size_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        organization_categorization: Mapped[str | None] = mapped_column(String, nullable=True)
        foundation_model: Mapped[str | None] = mapped_column(String, nullable=True)
        training_compute_lower_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
        training_compute_upper_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
        training_chip_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
        training_code_accessibility: Mapped[str | None] = mapped_column(String, nullable=True)
        accessibility_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        organization_categorization_from_organization: Mapped[str | None] = mapped_column(String, nullable=True)
        possibly_over_1e23_flop: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
        training_compute_cost_2023_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
        utilization_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        numerical_format: Mapped[str | None] = mapped_column(String, nullable=True)
        frontier_model: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
        training_power_draw_w: Mapped[float | None] = mapped_column(Float, nullable=True)
        training_compute_estimation_method: Mapped[str | None] = mapped_column(String, nullable=True)
        hugging_face_developer_id: Mapped[str | None] = mapped_column(String, nullable=True)
        post_training_compute_flop: Mapped[float | None] = mapped_column(Float, nullable=True)
        post_training_compute_notes: Mapped[str | None] = mapped_column(String, nullable=True)
        hardware_utilization_hfu: Mapped[float | None] = mapped_column(Float, nullable=True)


T_EpochTable = TypeVar("T_EpochTable", bound="AbstractEpochTable")


class AbstractEpochTable(EpochColumns, Base):
    __abstract__ = True

    column_mapping = COLUMN_MAPPING
    csv_to_attr_mapping = COLUMN_CSV_TO_ATTR
    date_columns = DATE_COLUMNS
    datetime_columns = DATETIME_COLUMNS
    integer_columns = INTEGER_COLUMNS
    float_columns = FLOAT_COLUMNS
    boolean_columns = BOOLEAN_COLUMNS

    def _convert_value(self, field: str, value: Any) -> Any:
        if pd.isna(value):  # type: ignore[reportUnknownMemberType]
            return None
        if field in self.date_columns:
            return to_date(value)
        if field in self.datetime_columns:
            return to_datetime(value)
        if field in self.boolean_columns:
            return to_boolean(value)
        if field in self.integer_columns:
            return to_int(value)
        if field in self.float_columns:
            return to_float(value)
        return value

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @classmethod
    def connect(cls: Type[T_EpochTable], engine: Engine) -> T_EpochTable:
        return cls(engine)

    @classmethod
    def create_table(cls: Type[T_EpochTable], engine: Engine) -> T_EpochTable:
        cls.metadata.create_all(engine, tables=[cast(Table, cls.__table__)])
        return cls(engine)


class EpochSplit(Base):
    __tablename__ = "epoch_split"

    id_paper: Mapped[int] = mapped_column(Integer, primary_key=True)
    split: Mapped[str] = mapped_column(String, nullable=False)  # one of {'train','dev','test'}


class EpochTable(AbstractEpochTable):
    __tablename__ = "epoch"


    def load_df(self) -> pd.DataFrame:
        os.makedirs(CACHE_DIR, exist_ok=True)
        zip_path = download_zip(EPOCH_TABLE_URL, CACHE_DIR / "epoch_table.zip")
        extract_zip(zip_path, CACHE_DIR / "epoch_table")
        df = pd.read_csv(CACHE_DIR / "epoch_table" / "all_ai_models.csv")  # type: ignore[reportUnknownMemberType]
        return df


    def load_table(self) -> None:
        df = self.load_df()
        if self.__tablename__ != "epoch":
            raise RuntimeError("load_table is only supported on the canonical epoch table")
        df = df.rename(columns=self.csv_to_attr_mapping)

        if "id_paper" not in df.columns:
            df["id_paper"] = range(1, len(df) + 1)

        for column in COLUMN_ORDER:
            if column not in df.columns:
                df[column] = pd.NA

        ordered_df = df[COLUMN_ORDER]
        ordered_df.to_sql("epoch", self.engine, if_exists="replace", index=False)  # type: ignore[reportUnknownMemberType]


    def create_splits(self, test_ratio: float, dev_ratio: float, *, random_state: int = 42) -> None:
        if test_ratio < 0 or dev_ratio < 0 or (test_ratio + dev_ratio) > 1.0:
            raise ValueError("Invalid ratios: require 0 <= test,dev and test+dev <= 1.0")
        train_ratio = 1.0 - test_ratio - dev_ratio

        df = pd.read_sql_table(self.__tablename__, self.engine)  # type: ignore[reportUnknownMemberType]
        df = normalize_epoch_dataframe(df)[["id_paper"]]
        df = cast(pd.DataFrame, df.sample(frac=1.0, random_state=random_state).reset_index(drop=True))  # type: ignore[reportUnknownMemberType]

        splits = shuffled_split(df, train_ratio, test_ratio, dev_ratio, random_state=random_state)

        # Prepare mapping records with split label
        records: list[dict[str, Any]] = []
        for label, frame in ("train", splits["train"]), ("test", splits["test"]), ("dev", splits["dev"]):
            for row in frame.to_dict(orient="records"):  # type: ignore[reportUnknownMemberType]
                records.append({"id_paper": int(row["id_paper"]), "split": label})

        split_table: Table = cast(Table, EpochSplit.__table__)
        with self.engine.begin() as connection:
            split_table.drop(connection, checkfirst=True)
            split_table.create(connection)

        if records:
            with self.engine.begin() as connection:
                connection.execute(split_table.insert(), records)



if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///data/epoch.db")
    try:
        epoch_table = EpochTable.connect(engine)
    except Exception as e:
        print(e)
        epoch_table = EpochTable.create_table(engine)
    epoch_table.load_table()
    epoch_table.create_splits(test_ratio=0.2, dev_ratio=0.1)
    