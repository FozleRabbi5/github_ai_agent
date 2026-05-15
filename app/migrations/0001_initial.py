from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="RepositoryIndex",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("repository_url", models.URLField(unique=True)),
                ("repository_name", models.CharField(max_length=255)),
                ("local_path", models.CharField(max_length=1024)),
                ("default_branch", models.CharField(blank=True, max_length=255)),
                ("last_commit_hash", models.CharField(blank=True, max_length=64)),
                (
                    "status",
                    models.CharField(
                        choices=[("pending", "Pending"), ("indexed", "Indexed"), ("failed", "Failed")],
                        default="pending",
                        max_length=32,
                    ),
                ),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("indexed_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["-updated_at"]},
        ),
        migrations.CreateModel(
            name="QuestionHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("thread_id", models.CharField(db_index=True, max_length=128)),
                ("question", models.TextField()),
                ("answer", models.TextField()),
                ("source_references", models.JSONField(blank=True, default=list)),
                ("execution_metadata", models.JSONField(blank=True, default=dict)),
                (
                    "status",
                    models.CharField(
                        choices=[("completed", "Completed"), ("failed", "Failed")],
                        default="completed",
                        max_length=32,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "repository",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="questions",
                        to="app.repositoryindex",
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
