

#ifndef YAML_UTIL_H
#define YAML_UTIL_H

#include "internal/util.h"

// The >> operator disappeared in yaml-cpp 0.5, so this function is
// added to provide support for code written under the yaml-cpp 0.3 API.
template<typename T>
void operator >> (const YAML::Node& node, T& i)
{
    i = node.as<T>();
}

template<typename ValueType>
void SetValueBasedOnYamlKey(const YAML::Node& node,
                              const std::string& key, ValueType* value) {
    if (!(node[key]
        && YAML::safeGet(node, key, value))) {
        LOG(ERROR) << "Unable to find " << key << ". So use " << *value;
    }
}

#endif // YAML_UTIL_H
