RM       := rm
CXX      ?= g++
CXXFLAGS := -pedantic -Wall -O3 -std=c++11
ADTFLAGS ?= -march=native -DWITH_PERF
LDFLAGS  :=
OBJDIR   := build
SOURCES  := $(shell find * -name "*.cpp" -type f -print)
OBJECTS  := $(SOURCES:%.cpp=$(OBJDIR)/%.o)
DEPS     := $(OBJECTS:$(OBJDIR)/%.o=$(OBJDIR)/%.d)
PROG     ?= gather
VERBOSE  ?= @

all: CXXFLAGS += $(ADTFLAGS)
all: $(PROG)

$(PROG): $(OBJECTS)
	@echo "LD		$@"
	$(VERBOSE) $(CXX) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	@if ! [ -d $(OBJDIR) ] ; then mkdir -p $(OBJDIR) ; fi
	@echo "CXX		$@"
	$(VERBOSE) $(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.d: %.cpp
	@if ! [ -d $(OBJDIR) ] ; then mkdir -p $(OBJDIR) ; fi
	@echo "DEP		$@"
	$(VERBOSE) $(CXX) $(CXXFLAGS) -MM -MF $@ -MT $(OBJDIR)/$*.o $<

clean:
	@echo "CLEAN"
	$(VERBOSE) $(RM) -rf build $(PROG) $(PROG)_*

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif

.PHONY: all clean

