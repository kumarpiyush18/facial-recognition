from flask_restful import Api

from flask import Flask, redirect, url_for, request, render_template, Response

from backend.api.view import HealthCheck, SearchIdentity

api = Api()

api.add_resource(HealthCheck, "/knockknock")
api.add_resource(SearchIdentity, "/search_identity")
# api.add_resource(MarkAttendance, "/mark_attendance")
