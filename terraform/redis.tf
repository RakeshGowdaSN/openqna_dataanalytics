data "google_compute_network" "redis_network" {
  name    = var.redis_network
  project = var.project_id
}

resource "google_vpc_access_connector" "redis" {
  count         = var.redis_enabled ? 1 : 0
  name          = "${var.cloud_run_service_name}-redis-connector"
  region        = var.deploy_region
  network       = data.google_compute_network.redis_network.name
  ip_cidr_range = var.vpc_connector_ip_cidr_range
  depends_on    = [module.project_services]
}

resource "google_redis_instance" "cache" {
  count              = var.redis_enabled ? 1 : 0
  name               = var.redis_name
  tier               = var.redis_tier
  memory_size_gb     = var.redis_memory_size_gb
  region             = var.redis_region
  authorized_network = data.google_compute_network.redis_network.self_link
  reserved_ip_range  = var.redis_reserved_ip_range
  depends_on         = [module.project_services]
}
