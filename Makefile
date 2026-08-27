# Convenience wrapper around docker/run.sh
IMAGE ?= braincomp:latest

.PHONY: help build selftest offline dry run phaseA phaseB shell clean-state clean-outputs

help:
	@echo "BrainCompensation - reproducible pipeline"
	@echo ""
	@echo "  make build         Build the Docker image ($(IMAGE))"
	@echo "  make selftest      Verify every dependency (touches no data)"
	@echo "  make offline       Same, with networking disabled (hermeticity check)"
	@echo "  make dry           Snakemake dry run - show the DAG"
	@echo "  make run           Run the full pipeline"
	@echo "  make phaseA        Phase A only (phenotypic + factor analysis)"
	@echo "  make phaseB        Phase B only (genetics / PGS)"
	@echo "  make shell         Interactive shell in the container"
	@echo "  make clean-state   Reset the container's snakemake state"
	@echo "  make clean-outputs Run the workflow's 'clean' rule (deletes results)"
	@echo ""
	@echo "  CORES=16 MEM_MB=64000 make run"

build:
	docker build --platform linux/amd64 -f docker/Dockerfile -t $(IMAGE) .

selftest:
	SKIP_BUILD=1 ./docker/run.sh --selftest

offline:
	SKIP_BUILD=1 DOCKER_RUN_EXTRA="--network none" ./docker/run.sh --selftest

dry:
	SKIP_BUILD=1 ./docker/run.sh -n

run:
	./docker/run.sh

phaseA:
	SKIP_BUILD=1 ./docker/run.sh /project/reports/A3_evaluate_social_factor_report.txt

phaseB:
	SKIP_BUILD=1 ./docker/run.sh /project/results/pgs_residuals.csv

shell:
	SKIP_BUILD=1 ./docker/run.sh bash

clean-state:
	rm -rf .docker/work/.snakemake

clean-outputs:
	SKIP_BUILD=1 ./docker/run.sh clean
