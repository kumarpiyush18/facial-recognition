from backend.database import db
from backend.models import ActivityLogs


def create_activity(member_id, entry_type):
    activity = ActivityLogs(
        member_id=member_id,
        entry_type=entry_type,
    )
    db.session.add(activity)
    db.session.commit()
