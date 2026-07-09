import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Configurar sys.path para importar correctamente el backend
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.main import app

def test_api_workflow():
    # Usar TestClient como context manager para ejecutar el evento startup
    with TestClient(app) as client:
        # 1. Probar Endpoint de Salud (/health)
        print("\n[*] Probando endpoint /health...")
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "ok"
        print(f"[+] Salud OK. Dispositivo detectado: {health_data['device']}")

        # Encontrar una imagen de prueba
        test_image_path = PROJECT_ROOT / "data" / "Imagen_Suelta" / "brazo_fractura.webp"
        if not test_image_path.exists():
            test_image_path = PROJECT_ROOT / "data" / "Imagen_Suelta" / "clavicula.jfif"
            
        assert test_image_path.exists(), "No se encontro ninguna imagen de prueba para ejecutar el test."
        print(f"[*] Usando imagen de prueba: {test_image_path}")

        # 2. Probar Endpoint de Prediccion (/predict)
        print("[*] Probando endpoint /predict...")
        with open(test_image_path, "rb") as image_file:
            predict_response = client.post(
                "/predict",
                files={"file": (test_image_path.name, image_file, "image/webp")},
                data={"threshold": "0.5"}
            )
        
        assert predict_response.status_code == 200, f"Error en /predict: {predict_response.text}"
        predict_data = predict_response.json()
        print(f"[+] Respuesta /predict exitosa: {predict_data}")
        assert "prediction" in predict_data
        assert "class_id" in predict_data
        assert "confidence" in predict_data
        assert "inference_time_ms" in predict_data
        assert predict_data["class_id"] in [0, 1]

        # 3. Probar Endpoint de Explicabilidad (/explain)
        print("[*] Probando endpoint /explain...")
        with open(test_image_path, "rb") as image_file:
            explain_response = client.post(
                "/explain",
                files={"file": (test_image_path.name, image_file, "image/webp")},
                data={"layer_name": "conv_head", "target_class": str(predict_data["class_id"])}
            )
        
        assert explain_response.status_code == 200, f"Error en /explain: {explain_response.text}"
        assert explain_response.headers["content-type"] == "image/png"
        assert len(explain_response.content) > 0, "La imagen devuelta por /explain esta vacia."
        print(f"[+] Respuesta /explain exitosa. Imagen PNG recibida ({len(explain_response.content)} bytes)")

        print("\n[OK] Todos los tests de API pasaron exitosamente!")

if __name__ == "__main__":
    # Permite ejecutar el archivo directamente con python
    test_api_workflow()
