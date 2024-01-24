from typing import Any

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Date, Integer, Numeric, DateTime, text

Base = declarative_base()


class Members(Base):
    __tablename__ = 'members'

    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    member_id = Column(String(200), primary_key=True)
    member_type = Column(String(200))
    first_name = Column(String(200))
    middle_name = Column(String(200))
    last_name = Column(String(200))
    abbr_name = Column(String(200))
    mentor_id = Column(String(200))
    dob = Column(String(200))
    gender = Column(String(1))  # Assuming gender is a single character

    def __init__(self, member_id, member_type, first_name, middle_name, last_name, abbr_name, mentor_id, dob, gender):
        self.member_id = member_id
        self.member_type = member_type
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.abbr_name = abbr_name
        self.mentor_id = mentor_id
        self.dob = dob
        self.gender = gender


class ActivityLogs(Base):
    __tablename__ = 'activity_logs'

    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    member_id = Column(String(200), primary_key=True)
    entry_at = Column(DateTime, nullable=False)
    exit_at = Column(DateTime, nullable=False)

    def __init__(self, member_id, entry_at, exit_at):
        self.member_id = member_id
        self.entry_at = entry_at
        self.exit_at = exit_at


class UserActivityLogs(Base):
    __tablename__ = 'user_activity_logs'

    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    year = Column(Integer, nullable=False)
    member_id = Column(String(200), primary_key=True)
    entry_map = Column(String(366))  # Assuming each entry_map has a length of 366 characters

    def __init__(self, year, member_id, entry_map):
        self.year = year
        self.member_id = member_id
        self.entry_map = entry_map
