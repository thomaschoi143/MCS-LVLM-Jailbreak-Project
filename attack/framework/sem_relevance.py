import requests
import logging
import commons.utils

logger = logging.getLogger("CustomLogger")


class SemRelevance:

    def compute_similarity(self, text1, text2):
        try:
            endpoint = "sem_relevance"
            response = requests.post(
                f"http://localhost:{commons.utils.PORT}/{endpoint}", json={"text1": text1, "text2": text2}
            )

            return response.json()["similarity"]
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        except requests.exceptions.ConnectionError:
            logging.error("Error: Could not connect to the server.")
        except requests.exceptions.Timeout:
            logging.error("Error: Request timed out.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
        except ValueError:
            logging.error("Error: Response is not valid JSON.")
        except Exception as e:
            logging.error(e)
        return None
