from flask_restful import Api

from flask import Flask, redirect, url_for, request, render_template, Response

from backend.api.view import HealthCheck

api = Api()

api.add_resource(HealthCheck, "/knockknock")


