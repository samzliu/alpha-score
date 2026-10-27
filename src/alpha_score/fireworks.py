"""Fireworks AI: endpoints and the on-demand deployment lifecycle (needed only for base models, which Fireworks does
not serve pay-per-token). Billed per GPU-second while up; ~$8/GPU-hour on H100/H200.
"""

import os
import time
import uuid

import requests

from .client import Client

API = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai/v1")  # overridable for verify/test_fireworks_mock.py
INFERENCE = os.environ.get("FIREWORKS_INFERENCE_BASE", "https://api.fireworks.ai/inference/v1")


class Fireworks:
    def __init__(self, key, account):
        self.key, self.account = key, account
        self.h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def req(self, method, path, **kw):
        r = requests.request(method, f"{API}/accounts/{self.account}{path}", headers=self.h, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Fireworks {method} {path}: {r.status_code} {r.text[:400]}")
        return r.json() if r.text else {}

    def create(self, base_model, accelerator=None, count=None, precision=None, dep_id=None):
        dep_id = dep_id or f"alpha-score-{uuid.uuid4().hex[:8]}"
        body = dict(baseModel=base_model, displayName=dep_id, minReplicaCount=0, maxReplicaCount=1,
                    # safety net: if this script dies before deleting, the replica still scales to zero
                    autoscalingPolicy=dict(scaleToZeroWindow="300s"))
        params = dict(deploymentId=dep_id)
        if accelerator or count or precision:  # explicit hardware = shapeless deployment
            body.update({k: v for k, v in dict(acceleratorType=accelerator, acceleratorCount=count, precision=precision).items() if v})
            params["skipShapeValidation"] = "true"
        else:
            body["deploymentShape"] = "default"  # let Fireworks pick a validated shape for the model
        self.req("POST", "/deployments", params=params, json=body)
        return dep_id

    def get(self, dep_id):
        return self.req("GET", f"/deployments/{dep_id}")

    def wait(self, dep_id, timeout_s=3600, log=print):
        t0 = time.time()
        while True:
            d = self.get(dep_id)
            state = d.get("state")
            if state == "READY":
                return d
            if state in ("FAILED", "DELETED", "DELETING"):
                raise RuntimeError(f"deployment {dep_id} is {state}: {d.get('status')}")
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"deployment {dep_id} still {state} after {timeout_s}s")
            log(f"  deployment {dep_id}: {state} ({int(time.time() - t0)}s)")
            time.sleep(20)

    def delete(self, dep_id):
        self.req("DELETE", f"/deployments/{dep_id}", params=dict(ignoreChecks="true"))

    def client(self, dep_id, base_model, top_k=True):
        """Inference client for a deployment; tries both documented model-string forms."""
        for model in (f"accounts/{self.account}/deployments/{dep_id}",
                      f"{base_model}#accounts/{self.account}/deployments/{dep_id}"):
            cl = Client(INFERENCE, model, self.key, top_k=top_k)
            try:
                cl.complete("Hello", retries=1, max_tokens=1)
                return cl
            except RuntimeError as e:  # 4xx: wrong model string
                last = e
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
                return cl  # right string, replica still scaling up: Client.wait_ready handles it
        raise RuntimeError(f"cannot address deployment {dep_id}: {last}")


def list_serverless(key):
    """Model ids Fireworks serves pay-per-token to this key."""
    r = requests.get(f"{API}/accounts/fireworks/models", headers={"Authorization": f"Bearer {key}"},
                     params=dict(pageSize=200, filter="supports_serverless=true"), timeout=60)
    r.raise_for_status()
    return sorted(x["name"] for x in r.json().get("models", []))
