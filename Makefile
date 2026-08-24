CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pthread -Iinc
LDFLAGS  := -pthread

SRC_DIR   := src
INC_DIR   := inc
EX_DIR    := examples
BUILD_DIR := build
BIN_DIR   := bin

# Core library source files and object files
LIB_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

# Automatic discovery of test drivers / examples
EX_SRCS := $(wildcard $(EX_DIR)/*.cpp)
EX_BINS := $(patsubst $(EX_DIR)/%.cpp,$(BIN_DIR)/%,$(EX_SRCS))

.PHONY: all clean run tsan

# Default target: builds all discovered executables
all: $(EX_BINS)

# Build with ThreadSanitizer enabled
# Forces a clean build to prevent mixing uninstrumented and instrumented object files
tsan: CXXFLAGS += -fsanitize=thread -g -O1
tsan: LDFLAGS  += -fsanitize=thread
tsan: clean all

# Compile core library object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile and link individual driver files from examples/ into bin/
$(BIN_DIR)/%: $(EX_DIR)/%.cpp $(LIB_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

# Create output directories if they don't exist
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Execute all discovered drivers sequentially
run: all
	@for bin in $(EX_BINS); do \
		echo "========================================"; \
		echo "Running test driver: $$bin"; \
		echo "========================================"; \
		./$$bin || exit 1; \
		echo ""; \
	done

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
