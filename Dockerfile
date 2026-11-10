# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Dashing2 v2.1.21, including its pinned recursive submodules.
WORKDIR /build/dashing2
RUN git init . \
    && git remote add origin https://github.com/dnbaker/dashing2.git \
    && git fetch --depth 1 origin adad4aa170829c1ef98c5195e075f33c0b73671f \
    && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive --depth 1

# Avoid -march=native: the image must also run on other CPUs of the same arch.
# Limit parallel compilation because Dashing2's C++ compilation uses substantial RAM.
ARG BUILD_JOBS=2
RUN make -j"${BUILD_JOBS}" dashing2 TARGET_FLAG= \
    && install -m 755 dashing2 /usr/local/bin/dashing2

WORKDIR /build/okkmer
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .
RUN mkdir -p /licenses/dashing2 \
    && cd /build/dashing2 \
    && find . -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' \) \
        -exec cp --parents '{}' /licenses/dashing2/ \;

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/dashing2 /usr/local/bin/dashing2
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels okkmer
COPY LICENSE /usr/local/share/licenses/okkmer/LICENSE
COPY --from=build /licenses/ /usr/local/share/licenses/

LABEL org.opencontainers.image.title="Ok-k-mer" \
      org.opencontainers.image.description="Select weighted ensembles of k-mer lengths against a reference phylogeny" \
      org.opencontainers.image.source="https://github.com/StephenHwang/Ok-k-mer" \
      org.opencontainers.image.licenses="GPL-3.0-only"

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1
WORKDIR /data
ENTRYPOINT ["okkmer"]
CMD ["--help"]

# Optional: docker build --target test -t okkmer:test .
FROM runtime AS test
COPY tests/ /tests/
RUN python -m unittest discover -s /tests -p 'test_*.py' -v

# Normal builds produce the runtime image without the test suite.
FROM runtime AS final
