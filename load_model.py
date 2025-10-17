{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOSjU28byaVZMbwNu3TMs1k"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fekw8LI70E0k"
      },
      "outputs": [],
      "source": [
        "import joblib\n",
        "import glob\n",
        "import os\n",
        "from typing import Any\n",
        "\n",
        "MODEL_DIR = os.environ.get(\"MODEL_DIR\", \"models\")\n",
        "\n",
        "def find_latest_model(pattern=\"mejor_modelo_*.joblib\"):\n",
        "    paths = glob.glob(os.path.join(MODEL_DIR, pattern))\n",
        "    if not paths:\n",
        "        raise FileNotFoundError(f\"No se encontró ningún modelo con patrón {pattern} en {MODEL_DIR}\")\n",
        "    # ordena por fecha de modificación y devuelve el más reciente\n",
        "    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)\n",
        "    return paths[0]\n",
        "\n",
        "def load_model(path: str = None) -> Any:\n",
        "    if path is None:\n",
        "        path = find_latest_model()\n",
        "    model = joblib.load(path)\n",
        "    return model\n"
      ]
    }
  ]
}