from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
import argparse
import datetime


# -----------------------------
# Parse input arguments
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription-id", dest="subscription_id", required=True)
    parser.add_argument("--resource-group", dest="resource_group", required=True)
    parser.add_argument("--workspace", dest="workspace", required=True)
    parser.add_argument("--endpoint-name", dest="endpoint_name", default="diabetes-endpoint")
    parser.add_argument("--deployment-name", dest="deployment_name", default="blue")

    return parser.parse_args()


# -----------------------------
# Connect to Azure ML
# -----------------------------
def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


# -----------------------------
# Create or get endpoint
# -----------------------------
def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print("Endpoint already exists ✅")
        return endpoint
    except Exception:
        print("Creating new endpoint...")

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Online endpoint for diabetes model",
            auth_mode="key",
        )

        return ml_client.begin_create_or_update(endpoint).result()


# -----------------------------
# Create deployment
# -----------------------------
def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
) -> ManagedOnlineDeployment:

    # ✅ Get existing registered model (IMPORTANT FIX)
    print("Fetching latest registered model...")
    model = ml_client.models.get(name="diabetes-model", label="latest")

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )

    print("Creating deployment...")
    return ml_client.online_deployments.begin_create_or_update(deployment).result()


# -----------------------------
# Route traffic
# -----------------------------
def set_traffic_to_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    print("Setting traffic to deployment...")
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.begin_create_or_update(endpoint).result()


# -----------------------------
# Main function
# -----------------------------
def main() -> None:
    args = parse_args()

    print("Connecting to Azure Machine Learning workspace...")
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    print(f"Ensuring endpoint '{args.endpoint_name}' exists...")
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)

    print(f"Using endpoint: {endpoint.name}")

    print(f"Creating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
    )

    print(f"Deployment state: {deployment.provisioning_state}")

    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)

    print("✅ Deployment complete!")
    print(f"Scoring URI: {endpoint.scoring_uri}")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    main()
