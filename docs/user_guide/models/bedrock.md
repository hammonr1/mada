# AWS Bedrock Model Configuration

This section provides details on configuring Bedrock models for use in the MADA Orchestrator.

## Fields

All fields, besides `extra`, can be either a literal value or an environment variable.

| Field Name            | Description                                           | Required? |
| --------------------- | ----------------------------------------------------- | --------- |
| `provider`            | The provider name, must be `aws-bedrock`.            | Yes       |
| `model`               | The name of the Bedrock model to use.                | Yes       |
| `region`              | AWS region for the Bedrock service.                  | Yes       |
| `bearer_token`        | Optional bearer token/API key value, or path to a file containing the token. | Conditionally        |
| `aws_profile`         | Optional AWS profile name.                           | Conditionally        |
| `aws_access_key_id`   | Optional AWS access key ID.                          | Conditionally        |
| `aws_secret_access_key` | Optional AWS secret access key.                    | Conditionally        |
| `aws_session_token`   | Optional AWS session token.                          | Optional        |
| `override_env_credentials` | Whether adapter setup may overwrite existing AWS credential-related environment variables. | No        |
| `extra`               | Additional settings (e.g., temperature, max_tokens). | No        |

## Authentication Requirements

A Bedrock model configuration must provide exactly **one** authentication method.

Supported authentication methods are:

| Authentication method         | Required fields |
| ----------------------------- | --------------- |
| Bearer token                  | `bearer_token` |
| AWS profile                   | `aws_profile` |
| Static AWS credentials        | `aws_access_key_id` and `aws_secret_access_key` |
| Session-based AWS credentials | `aws_access_key_id`, `aws_secret_access_key`, and optionally `aws_session_token` |

## Validation Rules

A Bedrock configuration is valid only if all of the following are true:

| Rule      | Requirement |
| --------- | ----------- |
| Provider  | `provider` must be `aws-bedrock` |
| Model     | `model` must be specified |
| Region    | `region` must be specified |
| Auth      | Exactly one [supported authentication method](#authentication-requirements) must be configured |

## Invalid Configurations

The following examples are invalid:

| Example | Why invalid |
| ------- | ----------- |
| `region` only | No authentication method provided |
| `bearer_token` and `aws_profile` | Multiple authentication methods provided |
| `aws_profile` and `aws_access_key_id` + `aws_secret_access_key` | Multiple authentication methods provided |
| `bearer_token` and `aws_access_key_id` + `aws_secret_access_key` | Multiple authentication methods provided |
| `aws_access_key_id` only | Missing `aws_secret_access_key` |
| `aws_secret_access_key` only | Missing `aws_access_key_id` |
| `aws_session_token` only | Missing access key ID and secret access key |
| `aws_session_token` with `aws_profile` only | Session token is only valid with static AWS credentials |

## Examples

!!! Example "Bearer Token Authentication"

    ```json
    "model": {
        "provider": "aws-bedrock",
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "region": "us-west-2",
        "bearer_token": "${AWS_BEARER_TOKEN_BEDROCK}"
    }
    ```

!!! Example "AWS Profile Authentication"

    ```json
    "model": {
        "provider": "aws-bedrock",
        "model": "amazon.nova-2-lite-v1:0",
        "region": "us-west-2",
        "aws_profile": "default"
    }
    ```

!!! Example "Static AWS Credentials Authentication"

    ```json
    "model": {
        "provider": "aws-bedrock",
        "model": "google.gemma-3-12b-it",
        "region": "us-west-2",
        "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
        "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}"
    }
    ```
