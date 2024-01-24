from typing import Any

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Date, Integer, Numeric, DateTime, text

Base = declarative_base()


class Members(Base):
    __tablename__ = 'members'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    deleted_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=True)
    member_id = Column(String(200), unique=True)
    member_type = Column(String(200)) # athlete, faculty, coach
    full_name = Column(String(200))
    abbr_name = Column(String(200))
    mentor_id = Column(String(200))
    sports = Column(String(200))
    gender = Column(String(1))  # Assuming gender is a single character

    # def __init__(self, member_id, member_type, full_name, abbr_name, mentor_id, gender):
    #     self.member_id = member_id
    #     self.member_type = member_type
    #     self.full_name = full_name
    #     self.abbr_name = abbr_name
    #     self.mentor_id = mentor_id
    #     self.gender = gender


class ActivityLogs(Base):
    __tablename__ = 'activity_logs'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    member_id = Column(String(200), nullable=False)
    entry_type = Column(String(200), nullable=False)
    # def __init__(self, member_id, entry_at, exit_at):
    #     self.member_id = member_id
    #     self.entry_at = entry_at
    #     self.exit_at = exit_at


class MemberActivityLogs(Base):
    __tablename__ = 'member_activity_logs'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    entry_at = Column(DateTime, nullable=False)
    exit_at = Column(DateTime, nullable=False)
    member_id = Column(String(200), nullable=False)

    # def __init__(self, year, member_id, entry_map):
    #     self.year = year
    #     self.member_id = member_id
    #     self.entry_map = entry_map
