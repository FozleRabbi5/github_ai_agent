from __future__ import annotations

import importlib.util
from typing import Any

from django.http import JsonResponse
from django.views.generic import TemplateView
from rest_framework import serializers
from rest_framework.schemas import get_schema_view
from rest_framework.schemas.openapi import AutoSchema


REQUIRED_OPENAPI_PACKAGES = ("inflection", "uritemplate")


def _schema_dependencies_available() -> bool:
    return all(importlib.util.find_spec(package) for package in REQUIRED_OPENAPI_PACKAGES)


class DocumentedAutoSchema(AutoSchema):
    def get_request_serializer(self, path: str, method: str):
        serializer_class = getattr(self.view, "request_serializer_class", None)
        return self._instantiate_serializer(serializer_class)

    def get_response_serializer(self, path: str, method: str):
        serializer_class = getattr(self.view, "response_serializer_class", None)
        many = getattr(self.view, "response_many", False)
        return self._instantiate_serializer(serializer_class, many=many)

    def get_filter_parameters(self, path: str, method: str):
        parameters = super().get_filter_parameters(path, method)
        parameters.extend(getattr(self.view, "schema_query_parameters", []))
        return parameters

    def get_components(self, path: str, method: str):
        components = super().get_components(path, method)
        response_serializer = self.get_response_serializer(path, method)
        if isinstance(response_serializer, serializers.ListSerializer):
            child = response_serializer.child
            components.setdefault(self.get_component_name(child), self.map_serializer(child))
        return components

    def get_responses(self, path: str, method: str):
        response_serializer = self.get_response_serializer(path, method)
        if isinstance(response_serializer, serializers.ListSerializer):
            self.response_media_types = self.map_renderers(path, method)
            item_schema = self.get_reference(response_serializer.child)
            return {
                "200": {
                    "content": {
                        content_type: {
                            "schema": {
                                "type": "array",
                                "items": item_schema,
                            }
                        }
                        for content_type in self.response_media_types
                    },
                    "description": "",
                }
            }
        return super().get_responses(path, method)

    def _instantiate_serializer(self, serializer_class: Any, many: bool = False):
        if serializer_class is None:
            return super().get_serializer("", "")
        if isinstance(serializer_class, (serializers.Serializer, serializers.ListSerializer)):
            return serializer_class
        return serializer_class(many=many)


def openapi_schema_view(request):
    if not _schema_dependencies_available():
        missing = [
            package for package in REQUIRED_OPENAPI_PACKAGES
            if importlib.util.find_spec(package) is None
        ]
        return JsonResponse(
            {
                "detail": (
                    "OpenAPI schema dependencies are missing. "
                    f"Install: {', '.join(missing)}"
                )
            },
            status=503,
        )

    schema_view = get_schema_view(
        title="GitHub AI Agent API",
        description=(
            "OpenAPI schema for the repository indexing and repository question-answering API."
        ),
        version="1.0.0",
        public=True,
    )
    return schema_view(request)


class SwaggerUIView(TemplateView):
    template_name = "app/swagger_ui.html"
    extra_context = {
        "schema_url": "/api/schema/",
        "title": "GitHub AI Agent Swagger UI",
        "missing_dependencies": not _schema_dependencies_available(),
        "required_packages": ", ".join(REQUIRED_OPENAPI_PACKAGES),
    }


class ReDocAPIView(TemplateView):
    template_name = "app/redoc.html"
    extra_context = {
        "schema_url": "/api/schema/",
        "title": "GitHub AI Agent ReDoc",
        "missing_dependencies": not _schema_dependencies_available(),
        "required_packages": ", ".join(REQUIRED_OPENAPI_PACKAGES),
    }
