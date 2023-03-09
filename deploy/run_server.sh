#!/bin/bash

uvicorn app.main:app --workers 4 --host=0.0.0.0 --port 11000