"""Tasks for maintaining the project."""

from invoke import task


@task
def create_environment(c):
    """Create a new environment for project."""
    c.run("uv venv")
    c.run("uv pip install -e .[dev]")


@task
def requirements(c):
    """Install project requirements."""
    c.run("uv pip install -e .")


@task
def dev_requirements(c):
    """Install development requirements."""
    c.run("uv pip install -e .[dev]")


@task
def preprocess_data(c):
    """Preprocess data."""
    c.run("uv run python src/mini_latent_pd/data.py")


@task
def train(c):
    """Train model."""
    c.run("uv run python src/mini_latent_pd/train.py")


@task
def test(c):
    """Run tests."""
    c.run("uv run pytest tests/")


@task
def build_docs(c):
    """Build documentation."""
    c.run("uv run mkdocs build")


@task
def serve_docs(c):
    """Serve documentation."""
    c.run("uv run mkdocs serve")


@task
def test_debug(c):
    """Run training with debug configuration for quick testing."""
    c.run("uv run python src/mini_latent_pd/train.py +experiment=debug")
