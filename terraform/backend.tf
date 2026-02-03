resource "google_firestore_database" "chathistory_db" {
  project     = var.project_id
  name        = "opendataqna-session-logs"
  location_id = var.firestore_region
  type        = "FIRESTORE_NATIVE"
  depends_on  = [ module.project_services ]
}

resource "google_cloud_run_service" "backend" {
  name     = var.cloud_run_service_name
  location = var.deploy_region
  project  = var.project_id

  template {
    spec {
      containers {
        image = "us-docker.pkg.dev/cloudrun/container/hello"
        dynamic "env" {
          for_each = var.redis_enabled ? {
            REDIS_HOST                 = google_redis_instance.cache[0].host
            REDIS_PORT                 = tostring(google_redis_instance.cache[0].port)
            REDIS_CACHE_ENABLED        = "true"
            REDIS_SSL                  = "false"
            CACHE_TTL_DEFAULT_SECONDS  = tostring(var.cache_ttl_default_seconds)
            CACHE_TTL_METADATA_SECONDS = tostring(var.cache_ttl_metadata_seconds)
            CACHE_TTL_SQL_SECONDS      = tostring(var.cache_ttl_sql_seconds)
            CACHE_TTL_RESULTS_SECONDS  = tostring(var.cache_ttl_results_seconds)
          } : {}
          content {
            name  = env.key
            value = env.value
          }
        }
      }
      dynamic "vpc_access" {
        for_each = var.redis_enabled ? [1] : []
        content {
          connector = google_vpc_access_connector.redis[0].name
          egress    = "ALL_TRAFFIC"
        }
      }
      service_account_name = module.genai_cloudrun_service_account.email
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [ module.project_services,null_resource.org_policy_temp, module.genai_cloudrun_service_account, local_file.config_ini,]
}
