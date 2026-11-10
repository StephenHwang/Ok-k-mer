# Running and publishing Ok-k-mer with Docker

Run these commands in the root of the Ok-k-mer repository. Docker Desktop must
be installed and running on macOS/Windows, or Docker Engine on Linux.
`docker version` should show both a Client and a Server section.

## 1. Build a local image

```bash
docker build --progress=plain -t okkmer:0.1.0 .
```

An **image** is the installed software package. Each `docker run` starts a
**container** from that image; `--rm` removes the container after it exits.
It does not remove the image or files written to a mounted host folder.

The Dockerfile compiles Dashing2 v2.1.21 from commit
`adad4aa170829c1ef98c5195e075f33c0b73671f` with its recursive submodules,
builds Python wheels, and copies the installed dependencies into a separate
runtime stage. The final image includes Python 3.11 and supports plotting without
a display server. It installs the local source checkout, including uncommitted
source changes. You do not need to push your code to GitHub before building.

The first build downloads dependencies and compiles C++; allow several minutes.
Later builds reuse cached layers. If compilation runs out of memory, use
`docker build --build-arg BUILD_JOBS=1 -t okkmer:0.1.0 .` or give Docker more RAM.
The base image and Python dependencies can receive updates; this is not a fully
locked environment. Record the pushed image's digest when reproducing a run.

## 2. Test the image

Check the CLI:

```bash
docker run --rm okkmer:0.1.0 --version
docker run --rm okkmer:0.1.0 help
docker run --rm okkmer:0.1.0 help index
```

Run the integration smoke test, including the **real bundled Dashing2**:

```bash
mkdir -p docker-test-output
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD/tests:/tests:ro" \
    -v "$PWD/docker-test-output:/data" \
    --entrypoint python okkmer:0.1.0 \
    /tests/docker_smoke.py /data/smoke-output
```

Expected final message: `Container smoke test passed.` The test generates five
small genomes, indexes k=6..8 using Dashing2, searches and annotates a tree, and
creates `heatmap.pdf` and `tree.svg`. Inspect these and the results tables in
`docker-test-output/smoke-output/`. For a repeat run, use another output directory
name; the script refuses to overwrite an existing directory.

Run the existing Python test suite in the container build environment:

```bash
docker build --target test --progress=plain -t okkmer:test .
```

A successful build means all unit tests passed. The ordinary image build does
not run or include the test suite. The unit tests use fake Dashing2 executables;
the separate smoke test above checks the actual bundled executable.

`--entrypoint` overrides the usual `okkmer` entrypoint. For example, to open a
shell for debugging:

```bash
docker run --rm -it --entrypoint /bin/sh okkmer:0.1.0
```

Dashing2 v2.1.21 prints its version with `dashing2 --version` but exits with
status 1; that behavior alone does not mean the installation failed.

## 3. Run your own analysis

Put `genomes.tsv`, your genome files, and `reference.nwk` in a working directory,
then run these commands from that directory. The container's working directory
is `/data`, mapped here to your current directory.

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" okkmer:0.1.0 \
    index --genomes genomes.tsv --k-low 6 --k-high 40 \
    --threads 8 --output my_index

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" okkmer:0.1.0 \
    search --tree reference.nwk --index my_index --global \
    --k-low 6 --k-high 40 --max-kmers 3 --threads 8 --output search_results

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" okkmer:0.1.0 \
    plot heatmap --results search_results --ensemble-size 2 --output heatmap.pdf
```

Relative paths in `genomes.tsv` remain relative to that manifest. Absolute host
paths such as `/Users/you/genomes/a.fna` will not work unless the same path is
mounted inside the container. Prefer `genomes/a.fna` within the mounted folder,
or mount another folder explicitly and use its container path in the manifest.
Outputs must be new or empty directories, as with the local CLI.

`--user` gives output files your host user/group IDs, which matters especially
on Linux. The examples use bash/zsh syntax; Windows PowerShell requires adjusted
mount syntax and can omit `--user`. Docker Desktop's CPU and memory allocation
limits the resources available to `--threads`.

## 4. Upload to Docker Hub

In [Docker Hub](https://hub.docker.com/), create a repository named `okkmer`
under your account (or organization). Choose public or private visibility.
Then replace `YOUR_DOCKERHUB_USERNAME` below with that namespace:

```bash
DOCKERHUB_USER=YOUR_DOCKERHUB_USERNAME
docker login
docker tag okkmer:0.1.0 "$DOCKERHUB_USER/okkmer:0.1.0"
docker push "$DOCKERHUB_USER/okkmer:0.1.0"
```

Complete the browser/device sign-in that `docker login` presents. Keep passwords
and access tokens out of your Dockerfile and repository. Docker's official
[build and share tutorial](https://docs.docker.com/get-started/tutorials/run-an-app/)
explains this login/tag/push flow.

Optionally also publish a moving `latest` tag:

```bash
docker tag okkmer:0.1.0 "$DOCKERHUB_USER/okkmer:latest"
docker push "$DOCKERHUB_USER/okkmer:latest"
```

Use a new version tag for subsequent releases rather than overwriting `0.1.0`.
An image built on Apple Silicon is normally **linux/arm64**; an image built on
an Intel/AMD machine is normally **linux/amd64**. A plain tag/push uploads only
the architecture you built. For typical x86 lab servers, use the next section.

After pushing, verify the remote image (on another machine if possible):

```bash
docker pull "$DOCKERHUB_USER/okkmer:0.1.0"
docker run --rm "$DOCKERHUB_USER/okkmer:0.1.0" --version
```

Private repositories require `docker login` on machines pulling the image.

## 5. Publish for both Apple Silicon and Intel/AMD

Docker Desktop includes Buildx and emulation. First build and test an AMD64
image locally on an Apple Silicon Mac (compiling under emulation can be slow):

```bash
docker buildx build --platform linux/amd64 --load -t okkmer:amd64 .
docker run --rm --platform linux/amd64 okkmer:amd64 --version
docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" \
    -v "$PWD/tests:/tests:ro" -v "$PWD/docker-test-output:/data" \
    --entrypoint python okkmer:amd64 \
    /tests/docker_smoke.py /data/smoke-output-amd64
```

Once both architecture builds pass their smoke tests, publish a combined image:

```bash
docker buildx create --name okkmer-builder --driver docker-container --use
docker buildx inspect --bootstrap
docker buildx build --platform linux/amd64,linux/arm64 \
    -t "$DOCKERHUB_USER/okkmer:0.1.0" \
    -t "$DOCKERHUB_USER/okkmer:latest" --push .
docker buildx imagetools inspect "$DOCKERHUB_USER/okkmer:0.1.0"
```

If `okkmer-builder` already exists, select it with
`docker buildx use okkmer-builder` instead of creating it again. This build pushes
directly to Docker Hub. The combined tag lets Docker pick the right architecture
when someone pulls it. See Docker's [multi-platform build guide](https://docs.docker.com/build/building/multi-platform/).
