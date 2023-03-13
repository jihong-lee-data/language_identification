#!/bin/bash


if [[ "$1" == "test" ]]; then
    uvicorn app.main:app --reload --host=0.0.0.0 --port=8000
else
    uvicorn app.main:app --workers 4 --host=0.0.0.0 --port=8000
fi

